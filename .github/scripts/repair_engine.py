"""Bounded, evidence-driven environment repair with independent re-verification."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from failure_classifier import classify  # noqa: E402
from environment_manager import EnvironmentManager  # noqa: E402
from pr_test_harness import run_environment_recovery  # noqa: E402

MAX_REPAIR_TIME_BUDGET_SECONDS = 1800
MAX_REPAIR_ATTEMPTS = 3


class RepairEngine:
    """Execute allowlisted Python environment repairs and re-verify every step."""

    def __init__(
        self,
        workspace,
        repository_root,
        detection: dict,
        verify_fn,
        max_attempts: int = 3,
        time_budget_seconds: int = 900,
        integrity_guard=None,
        sleep_fn=time.sleep,
        conversation=None,
        detect_fn=None,
        enable_agent_recovery: bool = True,
    ):
        self.workspace = workspace
        self.repository_root = repository_root
        self.detection = detection or {}
        self.verify_fn = verify_fn
        self.max_attempts = max(0, min(int(max_attempts), MAX_REPAIR_ATTEMPTS))
        self.time_budget_seconds = min(
            max(30, int(time_budget_seconds)), MAX_REPAIR_TIME_BUDGET_SECONDS
        )
        self.integrity_guard = integrity_guard
        self.sleep_fn = sleep_fn
        self.conversation = conversation
        self.detect_fn = detect_fn
        self.enable_agent_recovery = enable_agent_recovery
        self._agent_recovery_used = False
        self.manager = EnvironmentManager(workspace)
        self.deadline = time.monotonic() + self.time_budget_seconds

    def _remaining(self) -> float:
        return self.deadline - time.monotonic()

    def _integrity(self, stage: str) -> dict:
        if self.integrity_guard is None:
            return {"unchanged": True}
        result = self.integrity_guard()
        return result if isinstance(result, dict) else {"unchanged": False, "error": "invalid guard result"}

    def _fresh_detection(self, deadline: float) -> dict | None:
        """Re-read sandbox manifests before an install decision."""
        if self.detect_fn is None or self._remaining() <= 0:
            return None
        try:
            fresh = self.detect_fn()
        except Exception as exc:
            return {
                "schema_version": 1,
                "python": {"detected": False},
                "evidence": {"source": "sandbox", "error": f"{type(exc).__name__}: {exc}"},
            }
        return fresh if isinstance(fresh, dict) else None

    def _run_action(self, action: str, timeout: int, deadline: float) -> dict:
        if self._remaining() <= 0:
            return {
                "changed": False,
                "commands": [],
                "blocked": True,
                "reason": "repair deadline expired before action",
            }
        # Dependency/framework installs must use freshly detected sandbox state.
        if action in {"install_dependencies", "install_test_framework"}:
            fresh = self._fresh_detection(deadline)
            if fresh is not None:
                self.detection = fresh

        try:
            if action == "ensure_venv":
                result = self.manager.ensure_venv(
                    self.detection, timeout=timeout, deadline=deadline
                )
            elif action == "bootstrap_pip":
                result = self.manager.bootstrap_pip(timeout=timeout, deadline=deadline)
            elif action == "install_test_framework":
                result = self.manager.install_test_framework(
                    self.detection, timeout=timeout, deadline=deadline
                )
            elif action == "install_dependencies":
                result = self.manager.install_dependencies(
                    self.detection, timeout=timeout, deadline=deadline
                )
            elif action == "rebuild_plan":
                result = {
                    "changed": False,
                    "commands": [],
                    "reason": "plan rebuilt during independent verification",
                }
            else:
                result = {
                    "changed": False,
                    "commands": [],
                    "blocked": True,
                    "reason": f"unsupported repair action {action!r}",
                }
        except Exception as exc:
            result = {
                "changed": False,
                "commands": [],
                "blocked": True,
                "reason": f"{type(exc).__name__}: {exc}",
            }
        return result

    def _agent_recovery(self, current: dict) -> tuple[dict, dict]:
        """Use preserved agent recovery, but re-verify its result independently."""
        skipped = {
            "source": "agent",
            "result": "skipped",
            "reason": "agent recovery disabled or unavailable",
        }
        if not self.enable_agent_recovery or self.conversation is None:
            return current, skipped
        if self._agent_recovery_used:
            return current, {
                "source": "agent",
                "result": "blocked",
                "reason": "agent recovery already used; refusing a second turn",
            }
        if self._remaining() <= 0:
            return current, {"source": "agent", "result": "blocked", "reason": "repair deadline expired"}
        self._agent_recovery_used = True

        # Reserve part of the budget for integrity checks and the mandatory
        # trusted re-verification after the turn, including a timeout/cancel.
        remaining = self._remaining()
        reserve = min(30.0, max(5.0, remaining * 0.1))
        run_timeout = max(1.0, remaining - reserve)

        before_digest = current.get("plan_digest")
        try:
            # Bounded to one turn, integrity-guarded, and verified only through
            # the trusted verify_fn (never through agent claims).
            updated, attempts = run_environment_recovery(
                self.workspace,
                self.conversation,
                dict(current),
                self.verify_fn,
                max_attempts=1,
                integrity_guard=self.integrity_guard,
                sleep_fn=lambda _seconds: None,
                run_timeout=run_timeout,
            )
        except Exception as exc:
            return current, {
                "source": "agent",
                "result": "error",
                "reason": f"{type(exc).__name__}: {exc}",
            }

        if self._remaining() <= 0:
            return current, {
                "source": "agent",
                "result": "blocked",
                "reason": "repair deadline expired after agent recovery",
                "agent_recovery_attempts": attempts,
            }

        # The trusted controller is the final authority, even if the agent
        # recovery helper returned a 'ready' report.
        trusted = self.verify_fn()
        if not isinstance(trusted, dict):
            trusted = {
                "ready": False,
                "failures": ["agent recovery produced no verifiable report"],
                "checks": [],
                "execution_plan": None,
                "plan_digest": None,
            }
        entry = {
            "source": "agent",
            "result": "ready" if trusted.get("ready") else "still_blocked",
            "run_timeout_seconds": run_timeout,
            "agent_recovery_attempts": attempts,
            "plan_digest_before": before_digest,
            "plan_digest_after": trusted.get("plan_digest"),
            "plan_changed": before_digest != trusted.get("plan_digest"),
            "verification_ready": bool(trusted.get("ready")),
        }
        return trusted, entry

    def repair(self, verification: dict) -> tuple[dict, list[dict]]:
        """Repair until ready, blocked, or budget exhausted; always re-verify."""
        history: list[dict] = []
        current = verification
        seen_signatures: set[tuple] = set()

        for attempt in range(1, self.max_attempts + 1):
            if current.get("ready"):
                break
            remaining = self._remaining()
            if remaining <= 0:
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "reason": "environment repair time budget exhausted",
                })
                current = {
                    "ready": False,
                    "failures": ["environment repair time budget exhausted"],
                    "checks": [],
                    "execution_plan": None,
                    "plan_digest": None,
                }
                break

            classification = classify(current, self.detection)
            if not classification["repairable"]:
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "category": classification["category"],
                    "evidence": classification["evidence"],
                    "reason": classification["blocked_reason"],
                })
                break

            actions = list(classification["repair_actions"])
            signature = (
                classification["category"],
                tuple(actions),
                current.get("plan_digest"),
            )
            if signature in seen_signatures:
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "category": classification["category"],
                    "reason": "identical failed repair already attempted; refusing to retry unchanged work",
                    "evidence": classification["evidence"],
                })
                break
            seen_signatures.add(signature)

            before_integrity = self._integrity("before repair")
            if not before_integrity.get("unchanged"):
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "reason": "test integrity guard failed before repair",
                    "integrity": before_integrity,
                })
                break

            action_records = []
            blocked = False
            for action in actions:
                remaining = self._remaining()
                if remaining <= 0:
                    blocked = True
                    action_records.append({
                        "action": action,
                        "changed": False,
                        "commands": [],
                        "blocked": True,
                        "reason": "repair deadline expired before action",
                    })
                    break
                record = self._run_action(action, timeout=int(min(remaining, 300)), deadline=self.deadline)
                record = {"action": action, **record}
                action_records.append(record)
                if record.get("blocked"):
                    blocked = True
                    break

            after_integrity = self._integrity("after repair")
            if not after_integrity.get("unchanged"):
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "category": classification["category"],
                    "reason": "test integrity guard failed after repair",
                    "integrity": after_integrity,
                    "actions": action_records,
                })
                break

            if self._remaining() <= 0:
                history.append({
                    "attempt": attempt,
                    "result": "blocked",
                    "category": classification["category"],
                    "reason": "repair deadline expired before independent verification",
                    "actions": action_records,
                })
                current = {
                    "ready": False,
                    "failures": ["repair deadline expired before independent verification"],
                    "checks": [],
                    "execution_plan": None,
                    "plan_digest": None,
                }
                break

            # The trusted verifier is the only authority on whether the repair
            # worked. It rebuilds the plan from current sandbox state.
            try:
                current = self.verify_fn()
            except Exception as exc:
                current = {
                    "ready": False,
                    "failures": [f"independent re-verification raised {type(exc).__name__}: {exc}"],
                    "checks": [],
                    "execution_plan": None,
                    "plan_digest": None,
                }

            previous_digest = None
            for entry in reversed(history):
                if entry.get("plan_digest_after"):
                    previous_digest = entry["plan_digest_after"]
                    break
            if previous_digest is None:
                previous_digest = verification.get("plan_digest")
            new_digest = current.get("plan_digest")
            plan_changed = previous_digest != new_digest

            history.append({
                "attempt": attempt,
                "result": "ready" if current.get("ready") else ("blocked" if blocked else "reverified"),
                "category": classification["category"],
                "reason": classification["reason"],
                "evidence": classification["evidence"],
                "actions": action_records,
                "blocked": blocked,
                "integrity": after_integrity,
                "plan_digest_before": previous_digest,
                "plan_digest_after": new_digest,
                "plan_changed": plan_changed,
                "verification_ready": bool(current.get("ready")),
                "verification_failures": current.get("failures", []),
            })

            if current.get("ready") or blocked:
                break
            if attempt < self.max_attempts:
                self.sleep_fn(min(0.5 * (2 ** (attempt - 1)), 2.0))

        # Preserve the existing agent recovery path only as a bounded fallback;
        # its result is still re-verified by the trusted controller.
        if not current.get("ready") and self.enable_agent_recovery and self.conversation is not None:
            current, agent_entry = self._agent_recovery(current)
            history.append(agent_entry)

        current["environment_repair_history"] = history
        current["recovery_status"] = "READY" if current.get("ready") else "BLOCKED"
        return current, history


def configured_repair_limits() -> tuple[int, int]:
    """Read and clamp bounded retry settings."""
    attempts = min(max(0, int(os.getenv("MAX_PYTHON_ENVIRONMENT_REPAIRS", "3"))), MAX_REPAIR_ATTEMPTS)
    budget = min(
        max(30, int(os.getenv("ENVIRONMENT_REPAIR_TIME_BUDGET_SECONDS", "900"))),
        MAX_REPAIR_TIME_BUDGET_SECONDS,
    )
    return attempts, budget
