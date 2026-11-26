import importlib.util
import os
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / ".github/scripts/bootstrap_services.py"
SPEC = importlib.util.spec_from_file_location("bootstrap_services", SCRIPT)
bootstrap_services = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bootstrap_services)


class Result:
    def __init__(self, exit_code=0, stdout="", stderr=""):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr


def fake_runner(results=None):
    calls = []

    def runner(command, timeout=300, env=None):
        calls.append(command)
        if results:
            return results.pop(0)
        return Result()

    return runner, calls


def test_profile_validation_rejects_unknown_and_malformed_values(monkeypatch):
    assert bootstrap_services.load_profile("postgres_valkey_local")["db_name"] == "saleor"
    with pytest.raises(bootstrap_services.BootstrapError):
        bootstrap_services.load_profile("untrusted_profile")
    profile = dict(bootstrap_services.PROFILES["postgres_valkey_local"])
    profile["db_port"] = True
    monkeypatch.setitem(bootstrap_services.PROFILES, "bad_port", profile)
    with pytest.raises(bootstrap_services.BootstrapError):
        bootstrap_services.load_profile("bad_port")


def test_generated_env_uses_only_localhost_and_no_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("GITHUB_TOKEN", "host-github-token")
    monkeypatch.setenv("REGRESSION_TESTS_TOKEN", "host-regression-token")
    values = bootstrap_services.build_env(
        bootstrap_services.load_profile("postgres_valkey_local")
    )
    target = tmp_path / "services.env"
    bootstrap_services.write_env(target, values)

    content = target.read_text(encoding="utf-8")
    assert "host-github-token" not in content
    assert "host-regression-token" not in content
    assert "PASSWORD" not in content
    assert "@127.0.0.1" in content
    assert oct(target.stat().st_mode & 0o777) == "0o600"


def test_service_commands_never_touch_network_or_host_credentials():
    profile = bootstrap_services.load_profile("postgres_valkey_local")
    commands = (
        bootstrap_services.postgres_commands(profile, "secret")
        + bootstrap_services.cache_commands(profile)
        + bootstrap_services.readiness_commands(profile)
    )
    joined = "\n".join(commands)
    assert "127.0.0.1" in joined
    assert "iptables" not in joined
    assert "docker" not in joined
    assert "GITHUB_TOKEN" not in joined
    assert "REGRESSION_TESTS_TOKEN" not in joined


def test_bootstrap_starts_services_and_verifies(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap_services, "missing_executables", lambda: [])
    runner, calls = fake_runner()
    env_file = tmp_path / "services.env"

    result = bootstrap_services.bootstrap(
        "postgres_valkey_local", env_file, runner=runner
    )

    assert result == {
        "profile": "postgres_valkey_local",
        "env_file": str(env_file),
    }
    assert env_file.is_file()
    assert len(calls) == len(
        bootstrap_services.postgres_commands(
            bootstrap_services.load_profile("postgres_valkey_local"), "x"
        )
        + bootstrap_services.cache_commands(
            bootstrap_services.load_profile("postgres_valkey_local")
        )
        + bootstrap_services.readiness_commands(
            bootstrap_services.load_profile("postgres_valkey_local")
        )
    )


def test_bootstrap_fails_closed_on_service_command_error(monkeypatch, tmp_path):
    monkeypatch.setattr(bootstrap_services, "missing_executables", lambda: [])
    runner, _ = fake_runner([Result(exit_code=1, stderr="initdb failed")])
    with pytest.raises(bootstrap_services.BootstrapError, match="initdb failed"):
        bootstrap_services.bootstrap("postgres_valkey_local", tmp_path / "s.env", runner=runner)


def test_verify_rejects_incomplete_env_file(monkeypatch, tmp_path):
    profile = bootstrap_services.load_profile("postgres_valkey_local")
    env_file = tmp_path / "services.env"
    env_file.write_text("DATABASE_URL=postgres://x\n", encoding="utf-8")
    runner, calls = fake_runner()
    with pytest.raises(bootstrap_services.BootstrapError, match="incomplete"):
        bootstrap_services.verify(profile, env_file, runner=runner)
    assert calls == []


def test_install_packages_uses_proxy_environment(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    states = iter([["psql"], []])
    monkeypatch.setattr(
        bootstrap_services, "missing_executables", lambda: next(states)
    )
    runner, calls = fake_runner()
    profile = bootstrap_services.load_profile("postgres_valkey_local")

    bootstrap_services.install_packages(profile, runner=runner)

    assert len(calls) == 2
    assert calls[0] == "sudo apt-get update"
    assert calls[1].startswith("sudo apt-get install -y --no-install-recommends")
