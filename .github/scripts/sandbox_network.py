"""Create and tear down the isolated Docker network for untrusted sandboxes.

This runs on the trusted GitHub Actions host. It never handles PR-controlled
code and is invoked only from the test-generation workflow.
"""

from __future__ import annotations

import argparse
import os
import subprocess

NETWORK_NAME = "oh-pr-sandbox"
SUBNET = "172.28.0.0/16"
GATEWAY = "172.28.0.1"
PROXY_PORT = 3128
CHAIN = "OH_SANDBOX_GUARD"
METADATA = "169.254.169.254/32"


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        command,
        check=check,
        capture_output=True,
        text=True,
    )


def network_exists() -> bool:
    return run(["docker", "network", "inspect", NETWORK_NAME], check=False).returncode == 0


def create_network() -> None:
    if network_exists():
        return
    run([
        "docker", "network", "create",
        "--driver", "bridge",
        "--subnet", SUBNET,
        "--gateway", GATEWAY,
        "--opt", "com.docker.network.bridge.enable_icc=false",
        NETWORK_NAME,
    ])


def _iptables(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["sudo", "iptables", *args], check=check)


def _chain_exists() -> bool:
    return _iptables("-n", "-L", CHAIN, check=False).returncode == 0


def configure_firewall() -> None:
    """Apply fail-closed rules for the dedicated sandbox subnet."""
    if not _chain_exists():
        _iptables("-N", CHAIN)

    _iptables("-F", CHAIN)

    # Only apply to the dedicated sandbox subnet.
    _iptables("-A", CHAIN, "!", "-s", SUBNET, "-j", "RETURN")

    # Cloud metadata is never reachable.
    _iptables("-A", CHAIN, "-d", METADATA, "-j", "DROP")

    # Replies to the host-controlled agent-server channel must work.
    _iptables(
        "-A", CHAIN,
        "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED",
        "-j", "ACCEPT",
    )

    # The only new connection allowed to the gateway is the local package
    # proxy. Everything else to the gateway, internet, or other networks is
    # blocked.
    _iptables(
        "-A", CHAIN,
        "-d", GATEWAY, "-p", "tcp", "--dport", str(PROXY_PORT),
        "-j", "ACCEPT",
    )
    _iptables("-A", CHAIN, "-d", GATEWAY, "-j", "DROP")
    _iptables("-A", CHAIN, "-j", "DROP")

    for table_chain in ("DOCKER-USER", "INPUT"):
        exists = _iptables(
            "-C", table_chain, "-j", CHAIN, check=False
        ).returncode == 0
        if not exists:
            _iptables("-I", table_chain, "1", "-j", CHAIN)


def write_github_env(path: str) -> None:
    values = {
        "OH_SANDBOX_NETWORK": NETWORK_NAME,
        "OH_SANDBOX_SUBNET": SUBNET,
        "OH_SANDBOX_GATEWAY": GATEWAY,
        "PACKAGE_PROXY_URL": f"http://{GATEWAY}:{PROXY_PORT}",
        # Sandbox-scoped names: the generator maps these to HTTP_PROXY et al.
        # only while constructing the DockerWorkspace, so host-side LLM calls
        # never inherit the package proxy.
        "SANDBOX_HTTP_PROXY": f"http://{GATEWAY}:{PROXY_PORT}",
        "SANDBOX_HTTPS_PROXY": f"http://{GATEWAY}:{PROXY_PORT}",
        "SANDBOX_NO_PROXY": "127.0.0.1,localhost,169.254.169.254",
    }
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            handle.write(f"{key}={value}\n")


def _rules(table_chain: str) -> list[str]:
    result = _iptables("-S", table_chain, check=False)
    return [line for line in result.stdout.splitlines() if line.strip()]


def cleanup() -> None:
    """Remove only OH_SANDBOX_GUARD rules, then the chain and network."""
    watched = ("DOCKER-USER", "INPUT")
    before = {name: _rules(name) for name in watched}

    for table_chain in watched:
        while _iptables(
            "-C", table_chain, "-j", CHAIN, check=False
        ).returncode == 0:
            _iptables("-D", table_chain, "-j", CHAIN, check=False)

    after = {name: _rules(name) for name in watched}
    for name in watched:
        removed = [rule for rule in before[name] if rule not in after[name]]
        unrelated = [rule for rule in removed if CHAIN not in rule]
        if unrelated:
            raise RuntimeError(
                f"Cleanup touched unrelated rules in {name}: {unrelated}"
            )

    if _chain_exists():
        _iptables("-F", CHAIN, check=False)
        _iptables("-X", CHAIN, check=False)
    if _chain_exists():
        raise RuntimeError(f"Firewall chain {CHAIN} was not removed")
    run(["docker", "network", "rm", NETWORK_NAME], check=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configure", action="store_true")
    group.add_argument("--cleanup", action="store_true")
    parser.add_argument("--github-env", default=os.getenv("GITHUB_ENV"))
    args = parser.parse_args()

    if args.cleanup:
        cleanup()
        return

    create_network()
    configure_firewall()
    if args.github_env:
        write_github_env(args.github_env)
    print(f"Configured isolated sandbox network {NETWORK_NAME} on {SUBNET}")


if __name__ == "__main__":
    main()
