"""Tests for the trusted sandbox network setup and package proxy."""

import io
import os
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

import generate_tests as gt  # noqa: E402
import package_proxy as proxy  # noqa: E402
import sandbox_network as sn  # noqa: E402


def _record(monkeypatch, target, attribute="run"):
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(target, attribute, recorder)
    return calls


def test_create_network_uses_dedicated_ipv4_icc_disabled(monkeypatch):
    calls = []
    monkeypatch.setattr(sn, "network_exists", lambda: False)
    monkeypatch.setattr(
        sn, "run", lambda *a, **k: calls.append(a) or subprocess.CompletedProcess(a, 0)
    )

    sn.create_network()

    assert calls
    command = calls[0][0]
    assert command[:3] == ["docker", "network", "create"]
    assert "--driver" in command and "bridge" in command
    assert "--subnet" in command and sn.SUBNET in command
    assert "--gateway" in command and sn.GATEWAY in command
    assert "com.docker.network.bridge.enable_icc=false" in command
    assert "--ipv6" not in command
    assert "--privileged" not in command
    assert command[-1] == sn.NETWORK_NAME


def test_configure_firewall_installs_fail_closed_rules_and_hooks(monkeypatch):
    calls = _record(monkeypatch, sn, "_iptables")
    monkeypatch.setattr(sn, "_chain_exists", lambda: False)

    sn.configure_firewall()

    flat = [list(args) for args, _ in calls]
    assert ["-N", sn.CHAIN] in flat
    assert any("-d" in rule and sn.METADATA in rule and "DROP" in rule for rule in flat)
    assert any("ESTABLISHED,RELATED" in " ".join(rule) for rule in flat)
    assert any(
        "--dport" in rule and str(sn.PROXY_PORT) in rule and "ACCEPT" in rule
        for rule in flat
    )
    assert any("-d" in rule and sn.GATEWAY in rule and "DROP" in rule for rule in flat)
    assert any(rule[-1] == "DROP" and "-j" in rule for rule in flat)
    assert any("DOCKER-USER" in rule for rule in flat)
    assert any("INPUT" in rule for rule in flat)


def test_write_github_env_sets_network_and_proxy(tmp_path):
    env_file = tmp_path / "github_env"

    sn.write_github_env(str(env_file))

    content = env_file.read_text(encoding="utf-8")
    assert f"OH_SANDBOX_NETWORK={sn.NETWORK_NAME}" in content
    assert f"OH_SANDBOX_SUBNET={sn.SUBNET}" in content
    assert f"OH_SANDBOX_GATEWAY={sn.GATEWAY}" in content
    assert f"SANDBOX_HTTP_PROXY=http://{sn.GATEWAY}:{sn.PROXY_PORT}" in content
    assert f"SANDBOX_HTTPS_PROXY=http://{sn.GATEWAY}:{sn.PROXY_PORT}" in content
    lines = content.splitlines()
    assert f"HTTP_PROXY=http://{sn.GATEWAY}:{sn.PROXY_PORT}" not in lines
    assert f"HTTPS_PROXY=http://{sn.GATEWAY}:{sn.PROXY_PORT}" not in lines
    assert "169.254.169.254" in content


def test_cleanup_removes_only_guard_rules(monkeypatch):
    iptables_calls = []
    docker_calls = []
    seen_checks = set()
    chain_deleted = False

    rules = [
        f"-A DOCKER-USER -j {sn.CHAIN}",
        "-A DOCKER-USER -j RETURN",
        f"-A INPUT -j {sn.CHAIN}",
        "-P INPUT ACCEPT",
    ]

    def iptables(*args, **kwargs):
        nonlocal chain_deleted
        iptables_calls.append(args)
        if args[0] == "-C":
            key = args[1]
            if key in seen_checks:
                return subprocess.CompletedProcess(args, 1)
            seen_checks.add(key)
            return subprocess.CompletedProcess(args, 0)
        if args[0] == "-S":
            remaining = [r for r in rules if sn.CHAIN not in r] if chain_deleted else rules
            return subprocess.CompletedProcess(args, 0, stdout="\n".join(remaining))
        if args[0] == "-X":
            chain_deleted = True
        return subprocess.CompletedProcess(args, 0)

    def docker(*args, **kwargs):
        docker_calls.append(args)
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(sn, "_iptables", iptables)
    monkeypatch.setattr(sn, "run", docker)
    monkeypatch.setattr(sn, "_chain_exists", lambda: not chain_deleted)

    sn.cleanup()

    assert any(args[0] == "-D" and "DOCKER-USER" in args for args in iptables_calls)
    assert any(args[0] == "-D" and "INPUT" in args for args in iptables_calls)
    assert any(args[0] == "-X" for args in iptables_calls)
    assert any("network" in args[0] and "rm" in args[0] for args in docker_calls)


def test_generator_maps_and_restores_sandbox_proxy_env(monkeypatch):
    monkeypatch.setenv("HTTP_PROXY", "http://original-host-proxy:8080")
    monkeypatch.setenv("SANDBOX_HTTP_PROXY", "http://172.28.0.1:3128")
    monkeypatch.setenv("SANDBOX_NO_PROXY", "127.0.0.1,localhost")

    values, originals = gt.take_sandbox_proxy_env()

    assert values["HTTP_PROXY"] == "http://172.28.0.1:3128"
    assert originals["HTTP_PROXY"] == "http://original-host-proxy:8080"
    assert "SANDBOX_HTTP_PROXY" not in os.environ
    assert "HTTP_PROXY" not in os.environ

    gt.expose_sandbox_proxy_env(values)
    assert os.environ["HTTP_PROXY"] == "http://172.28.0.1:3128"

    gt.restore_sandbox_proxy_env(originals)
    assert os.environ["HTTP_PROXY"] == "http://original-host-proxy:8080"
    assert "HTTPS_PROXY" not in os.environ


def test_generator_requires_explicit_sandbox_network():
    source = (
        Path(__file__).resolve().parents[1] / ".github/scripts/generate_tests.py"
    ).read_text(encoding="utf-8")
    assert "OH_SANDBOX_NETWORK is required" in source
    assert 'network=sandbox_network' in source
    assert 'network="bridge"' not in source


def test_allowed_host_is_exact_or_subdomain_only():
    allowlist = ("pypi.org", "files.pythonhosted.org")
    assert proxy.allowed_host("pypi.org", allowlist)
    assert proxy.allowed_host("FILES.PYTHONHOSTED.ORG", allowlist)
    assert not proxy.allowed_host("evilpypi.org", allowlist)
    assert not proxy.allowed_host("pypi.org.evil.test", allowlist)


def test_public_ip_check_rejects_internal_and_metadata_addresses():
    for address in (
        "127.0.0.1",
        "10.0.0.1",
        "172.17.0.1",
        "169.254.169.254",
        "192.168.1.1",
        "::1",
    ):
        assert not proxy.is_public_ip(address), address
    assert proxy.is_public_ip("151.101.0.223")


def test_is_public_ip_rejects_cgnat_and_non_global_ranges():
    for address in ("100.64.0.1", "192.0.0.1", "198.18.0.1", "169.254.169.254"):
        assert not proxy.is_public_ip(address), address
    assert proxy.is_public_ip("151.101.0.223")


def test_package_proxy_rejects_oversized_request_body(monkeypatch):
    handler = object.__new__(proxy.AllowlistProxyHandler)
    handler.path = "http://pypi.org/simple/"
    handler.headers = {"Content-Length": str(proxy.MAX_BODY_BYTES + 1)}
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    rejected = []
    handler._reject = lambda status="502 Bad Gateway": rejected.append(status)

    handler._handle_absolute("POST")

    assert any("413" in status for status in rejected)
    assert handler.rfile.tell() == 0


def test_package_proxy_limits_concurrent_connections(monkeypatch):
    server = object.__new__(proxy.ThreadingProxy)
    server._slots = proxy.threading.BoundedSemaphore(0)
    monkeypatch.setattr(
        proxy.threading,
        "Thread",
        lambda **kwargs: SimpleNamespace(start=lambda: None),
    )

    class Request:
        closed = False

        def close(self):
            self.closed = True

    request = Request()
    server.process_request(request, ("127.0.0.1", 12345))

    assert request.closed is True


def test_workflow_verifies_ipv6_isolation():
    workflow = (
        Path(__file__).resolve().parents[1]
        / ".github/workflows/openhands-test-generation.yml"
    ).read_text(encoding="utf-8")

    assert "/proc/net/if_inet6" in workflow
    assert "IPv6 address probe failed" in workflow
    assert "IPv6 egress probe failed" in workflow
    assert "sandbox_network.py --cleanup || true" not in workflow


def test_block_embedded_dns_installs_netns_rules_and_verifies(monkeypatch):
    calls = []
    commands = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(args, 0, stdout="4242\n")
        return subprocess.CompletedProcess(args, 0)

    monkeypatch.setattr(gt.subprocess, "run", fake_run)
    workspace = SimpleNamespace(
        _container_id="container-abc",
        execute_command=lambda command: commands.append(command) or SimpleNamespace(
            exit_code=0, stdout="", stderr=""
        ),
    )

    gt.block_embedded_dns(workspace)

    assert any("nsenter" in call for call in calls)
    assert any("iptables" in call for call in calls if "nsenter" in call)
    assert any("ip6tables" in call for call in calls if "nsenter" in call)
    assert any("127.0.0.11" in " ".join(call) for call in calls)
    assert any("::1" in " ".join(call) for call in calls)
    assert commands == [gt.DNS_UDP_PROBE_COMMAND, gt.DNS_TCP_PROBE_COMMAND]
    assert "socket.create_connection" in gt.DNS_TCP_PROBE_COMMAND


def test_block_embedded_dns_fails_closed_when_probe_still_answers(monkeypatch):
    monkeypatch.setattr(
        gt.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(
            args, 0, stdout="4242\n"
        ),
    )
    workspace = SimpleNamespace(
        _container_id="container-abc",
        execute_command=lambda command: SimpleNamespace(exit_code=1, stdout="", stderr=""),
    )

    import pytest

    with pytest.raises(RuntimeError, match="still reachable"):
        gt.block_embedded_dns(workspace)


def test_resolve_container_id_supports_known_attributes():
    assert gt.resolve_container_id(SimpleNamespace(_container_id=" abc ")) == "abc"
    assert gt.resolve_container_id(SimpleNamespace(container_id="xyz")) == "xyz"


def test_resolve_container_id_fails_closed_when_missing():
    with pytest.raises(RuntimeError, match="container id is unavailable"):
        gt.resolve_container_id(SimpleNamespace())


def test_block_embedded_dns_rejects_non_numeric_pid(monkeypatch):
    monkeypatch.setattr(
        gt.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, stdout="not-a-pid\n"),
    )
    workspace = SimpleNamespace(
        _container_id="container-abc",
        execute_command=lambda command: SimpleNamespace(exit_code=0),
    )

    with pytest.raises(RuntimeError, match="Could not resolve sandbox container PID"):
        gt.block_embedded_dns(workspace)


def _start_proxy():
    server = proxy.ThreadingProxy(("127.0.0.1", 0), proxy.AllowlistProxyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, server.server_address[1]


def _proxy_request(port: int, raw: bytes) -> bytes:
    client = socket.create_connection(("127.0.0.1", port), timeout=3)
    client.sendall(raw)
    chunks = []
    while True:
        try:
            chunk = client.recv(4096)
        except socket.timeout:
            break
        if not chunk:
            break
        chunks.append(chunk)
    client.close()
    return b"".join(chunks)


def test_proxy_connect_rejects_disallowed_host():
    server, port = _start_proxy()
    try:
        response = _proxy_request(
            port,
            b"CONNECT evil.example:443 HTTP/1.1\r\nHost: evil.example:443\r\n\r\n",
        )
        assert b"403" in response.splitlines()[0]
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_connect_rejects_private_resolved_ip(monkeypatch):
    monkeypatch.setattr(proxy, "resolve_public", lambda host: "127.0.0.1")
    server, port = _start_proxy()
    try:
        response = _proxy_request(
            port,
            b"CONNECT pypi.org:443 HTTP/1.1\r\nHost: pypi.org:443\r\n\r\n",
        )
        assert b"403" in response.splitlines()[0]
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_get_forwards_allowed_absolute_request(monkeypatch):
    upstream = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    upstream.bind(("127.0.0.1", 0))
    upstream.listen(1)
    upstream_address = upstream.getsockname()

    def serve_upstream():
        connection, _ = upstream.accept()
        connection.recv(65536)
        connection.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
        connection.close()
        upstream.close()

    threading.Thread(target=serve_upstream, daemon=True).start()
    monkeypatch.setattr(
        proxy.AllowlistProxyHandler,
        "_connect",
        lambda self, host, port: socket.create_connection(upstream_address, timeout=3),
    )

    server, port = _start_proxy()
    try:
        response = _proxy_request(
            port,
            b"GET http://pypi.org/simple/ HTTP/1.1\r\nHost: pypi.org\r\n\r\n",
        )
        assert b"200 OK" in response
        assert response.endswith(b"ok")
    finally:
        server.shutdown()
        server.server_close()


def test_proxy_rejects_oversized_request_over_real_socket():
    server, port = _start_proxy()
    try:
        request = (
            b"POST http://pypi.org/simple/ HTTP/1.1\r\n"
            b"Host: pypi.org\r\n"
            + f"Content-Length: {proxy.MAX_BODY_BYTES + 1}\r\n".encode()
            + b"\r\n"
        )
        response = _proxy_request(port, request)
        assert b"413" in response.splitlines()[0]
    finally:
        server.shutdown()
        server.server_close()


def test_resolve_public_rejects_any_private_answer(monkeypatch):
    public = (2, 1, 6, "", ("151.101.0.223", 0))

    monkeypatch.setattr(proxy.socket, "getaddrinfo", lambda *a, **k: [public])
    assert proxy.resolve_public("pypi.org") == "151.101.0.223"

    private = (2, 1, 6, "", ("10.0.0.5", 0))
    monkeypatch.setattr(proxy.socket, "getaddrinfo", lambda *a, **k: [public, private])
    assert proxy.resolve_public("pypi.org") is None
