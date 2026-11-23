"""Minimal allowlisting HTTP/HTTPS proxy for sandbox package installation.

Only the domains required to install Python/Node/system packages are allowed.
Targets resolving to private, loopback, link-local, or otherwise non-public
addresses are rejected, including the cloud metadata endpoint.
"""

from __future__ import annotations

import argparse
import ipaddress
import select
import socket
import socketserver
import threading
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlsplit

MAX_BODY_BYTES = 1024 * 1024
MAX_HEADERS = 100
MAX_CONNECTIONS = 64

DEFAULT_ALLOWLIST = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "registry.yarnpkg.com",
    "deb.debian.org",
    "security.debian.org",
    "archive.ubuntu.com",
    "security.ubuntu.com",
)

HOP_BY_HOP = {
    "connection",
    "proxy-connection",
    "proxy-authenticate",
    "proxy-authorization",
    "keep-alive",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def allowed_host(host: str, allowlist: tuple[str, ...]) -> bool:
    host = host.lower().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in allowlist)


def is_public_ip(address: str) -> bool:
    """Return True only for globally routable, non-special addresses."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.is_global and not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def resolve_public(host: str) -> str | None:
    """Return a single public IP for host, or None if any result is unsafe."""
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except OSError:
        return None
    addresses = [info[4][0] for info in infos]
    if not addresses or not all(is_public_ip(address) for address in addresses):
        return None
    return addresses[0]


def _copy_socket(source: socket.socket, destination: socket.socket) -> None:
    try:
        while True:
            readable, _, _ = select.select([source], [], [], 30)
            if not readable:
                break
            chunk = source.recv(65536)
            if not chunk:
                break
            destination.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            destination.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class AllowlistProxyHandler(BaseHTTPRequestHandler):
    allowlist: tuple[str, ...] = DEFAULT_ALLOWLIST

    def log_message(self, format: str, *args) -> None:
        """Suppress request logging; URLs may contain sensitive query data."""
        return

    def _reject(self, status: str = "502 Bad Gateway") -> None:
        self.wfile.write(
            f"HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: 0\r\n\r\n".encode()
        )

    def _connect(self, host: str, port: int) -> socket.socket | None:
        if not allowed_host(host, self.allowlist):
            return None
        if port not in (80, 443):
            return None
        address = resolve_public(host)
        if address is None:
            return None
        try:
            return socket.create_connection((address, port), timeout=15)
        except OSError:
            return None

    def do_CONNECT(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        try:
            host, _, port_text = self.path.rpartition(":")
            port = int(port_text)
        except ValueError:
            self._reject("400 Bad Request")
            return
        upstream = self._connect(host, port)
        if upstream is None:
            self._reject("403 Forbidden")
            return
        self.wfile.write(b"HTTP/1.1 200 Connection established\r\n\r\n")
        self.wfile.flush()
        client = self.connection
        forwarder = threading.Thread(
            target=_copy_socket, args=(client, upstream), daemon=True
        )
        forwarder.start()
        _copy_socket(upstream, client)
        try:
            upstream.close()
        except OSError:
            pass

    def _handle_absolute(self, method: str) -> None:
        target = urlsplit(self.path)
        if target.scheme != "http" or not target.hostname:
            self._reject("400 Bad Request")
            return
        if len(self.headers) > MAX_HEADERS:
            self._reject("431 Request Header Fields Too Large")
            return
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._reject("400 Bad Request")
            return
        if length < 0 or length > MAX_BODY_BYTES:
            self._reject("413 Payload Too Large")
            return
        upstream = self._connect(target.hostname, target.port or 80)
        if upstream is None:
            self._reject("403 Forbidden")
            return
        headers = []
        body = self.rfile.read(length) if length else b""
        for key, value in self.headers.items():
            if key.lower() not in HOP_BY_HOP:
                headers.append(f"{key}: {value}")
        path = target.path or "/"
        if target.query:
            path = f"{path}?{target.query}"
        request = (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: {target.hostname}\r\n"
            + "\r\n".join(headers)
            + "\r\n\r\n"
        ).encode() + body
        upstream.sendall(request)
        while True:
            chunk = upstream.recv(65536)
            if not chunk:
                break
            self.wfile.write(chunk)
            self.wfile.flush()
        upstream.close()

    def do_GET(self) -> None:  # noqa: N802
        self._handle_absolute("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle_absolute("POST")

    def do_HEAD(self) -> None:  # noqa: N802
        self._handle_absolute("HEAD")


class ThreadingProxy(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slots = threading.BoundedSemaphore(MAX_CONNECTIONS)

    def process_request(self, request, client_address):
        if not self._slots.acquire(blocking=False):
            request.close()
            return
        thread = threading.Thread(
            target=self._process_request_with_slot,
            args=(request, client_address),
            daemon=True,
        )
        thread.start()

    def _process_request_with_slot(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        finally:
            self.shutdown_request(request)
            self._slots.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3128)
    parser.add_argument("--allowlist", default=",".join(DEFAULT_ALLOWLIST))
    args = parser.parse_args()

    allowlist = tuple(
        item.strip().lower()
        for item in args.allowlist.split(",")
        if item.strip()
    )
    handler = type(
        "ConfiguredAllowlistProxyHandler",
        (AllowlistProxyHandler,),
        {"allowlist": allowlist},
    )
    server = ThreadingProxy((args.host, args.port), handler)
    print(f"Allowlisting proxy on {args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
