#!/usr/bin/env python3
"""Bootstrap temporary, localhost-only services inside the OpenHands sandbox.

This script is trusted automation. It is uploaded from the base-branch checkout
and executed before the untrusted agent starts. It never reads host credentials
and never changes the host firewall or the isolated Docker network.
"""

import argparse
import json
import os
import re
import secrets
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


PROFILES = {
    "postgres_valkey_local": {
        "packages": [
            "postgresql", "redis-server", "libpq5", "libmagic1",
            "libxslt1-dev", "libcurl4",
        ],
        "data_dir": "/tmp/oh-services/pgdata",
        "log_dir": "/tmp/oh-services",
        "db_port": 5432,
        "cache_port": 6379,
        "db_name": "saleor",
        "db_user": "saleor",
    },
}

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class BootstrapError(RuntimeError):
    """Raised when local services cannot be prepared or verified."""


def execute(command, *, timeout=600, env=None):
    """Run one shell command with captured output (monkeypatched in tests)."""
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )
    return SimpleNamespace(
        exit_code=result.returncode,
        stdout=result.stdout or "",
        stderr=result.stderr or "",
    )


def load_profile(name):
    """Return a validated, trusted service profile."""
    profile = PROFILES.get(name)
    if not isinstance(profile, dict):
        raise BootstrapError(f"Unknown service profile: {name}")
    for key in ("db_name", "db_user"):
        if not IDENTIFIER_RE.fullmatch(str(profile.get(key, ""))):
            raise BootstrapError(f"Invalid profile identifier: {key}")
    for key in ("db_port", "cache_port"):
        value = profile.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or not (1 <= value <= 65535):
            raise BootstrapError(f"Invalid profile port: {key}")
    packages = profile.get("packages")
    if not isinstance(packages, list) or not packages or any(
        not isinstance(item, str) or not re.fullmatch(r"[a-z0-9][a-z0-9.+-]*", item)
        for item in packages
    ):
        raise BootstrapError("Invalid package list in service profile")
    for key in ("data_dir", "log_dir"):
        value = profile.get(key)
        if not isinstance(value, str) or not value.startswith("/tmp/oh-services"):
            raise BootstrapError(f"Invalid profile directory: {key}")
    return dict(profile)


def apt_environment():
    """Use only proxy variables already forwarded by trusted automation."""
    env = dict(os.environ)
    env["DEBIAN_FRONTEND"] = "noninteractive"
    for upper in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
        lower = upper.lower()
        if upper in env:
            env[lower] = env[upper]
    return env


def required_executables():
    """Executables that must exist after package installation."""
    return ("psql", "redis-server")


def missing_executables():
    return [name for name in required_executables() if not shutil.which(name)]


def install_packages(profile, runner=execute):
    """Install service packages through the sandbox-only proxy."""
    if not missing_executables():
        return
    env = apt_environment()
    commands = [
        "sudo apt-get update",
        "sudo apt-get install -y --no-install-recommends "
        + " ".join(shlex.quote(item) for item in profile["packages"]),
    ]
    for command in commands:
        result = runner(command, timeout=600, env=env)
        if result.exit_code != 0:
            raise BootstrapError(
                "Service package installation failed: "
                + (result.stderr or result.stdout or "").strip()[:2000]
            )
    still_missing = missing_executables()
    if still_missing:
        raise BootstrapError("Services unavailable after installation: " + ", ".join(still_missing))


def build_env(profile, password=None):
    """Build sandbox-local connection settings without storing any secret.

    The temporary cluster uses localhost-only trust authentication inside the
    disposable container. No password is written to the env file, so test
    output and reports cannot leak one.
    """
    user = profile["db_user"]
    name = profile["db_name"]
    port = profile["db_port"]
    cache = profile["cache_port"]
    return {
        "DATABASE_URL": f"postgres://{user}@127.0.0.1:{port}/{name}",
        "DATABASE_URL_REPLICA": f"postgres://{user}@127.0.0.1:{port}/{name}",
        "CACHE_URL": f"redis://127.0.0.1:{cache}/0",
        "CELERY_BROKER_URL": f"redis://127.0.0.1:{cache}/1",
    }


def write_env(path, values):
    """Write a mode-0600 env file containing only sandbox-local settings."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{key}={value}" for key, value in sorted(values.items())]
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    target.chmod(0o600)


def read_env(path):
    """Read the sandbox-local env file without exposing values in logs."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value
    return values


def postgres_commands(profile, password):
    """Return trusted, localhost-only PostgreSQL commands."""
    data = shlex.quote(profile["data_dir"])
    logs = shlex.quote(profile["log_dir"])
    user = shlex.quote(profile["db_user"])
    name = shlex.quote(profile["db_name"])
    port = str(profile["db_port"])
    secret = shlex.quote(password)
    base = f"sudo -u postgres sh -c"
    init = (
        f'{base} \'initdb_bin=$(ls -d /usr/lib/postgresql/*/bin/initdb | head -1); '
        f'"$initdb_bin" -D {data} -A trust --no-locale\''
    )
    start = (
        f'{base} \'pg_ctl_bin=$(ls -d /usr/lib/postgresql/*/bin/pg_ctl | head -1); '
        f'"$pg_ctl_bin" -D {data} '
        f'-o "-c listen_addresses=127.0.0.1 -p {port}" '
        f'-l {logs}/postgres.log start\''
    )
    role = (
        f"psql -h 127.0.0.1 -p {port} -U postgres -d postgres -v ON_ERROR_STOP=1 "
        f'-c "CREATE ROLE {user} LOGIN CREATEDB PASSWORD {secret}"'
    )
    database = (
        f"psql -h 127.0.0.1 -p {port} -U postgres -d postgres -v ON_ERROR_STOP=1 "
        f'-c "CREATE DATABASE {name} OWNER {user}"'
    )
    return [
        f"sudo install -d -o postgres -g postgres -m 700 {data} {logs}",
        init,
        start,
        role,
        database,
    ]


def cache_commands(profile):
    """Return trusted, localhost-only Valkey/Redis commands."""
    port = str(profile["cache_port"])
    logs = shlex.quote(profile["log_dir"])
    return [
        f"redis-server --bind 127.0.0.1 --port {port} --daemonize yes "
        f'--save "" --appendonly no --logfile {logs}/cache.log'
    ]


def readiness_commands(profile):
    """Return commands that only probe localhost services."""
    return [
        "python -c 'import socket,sys; "
        f"s=socket.create_connection((\"127.0.0.1\",{profile['db_port']}),2); "
        "s.close(); print(\"database reachable\")'",
        "python -c 'import socket,sys; "
        f"s=socket.create_connection((\"127.0.0.1\",{profile['cache_port']}),2); "
        "s.close(); print(\"cache reachable\")'",
    ]


def bootstrap(profile_name, env_file, runner=execute):
    """Install, start and verify temporary local services."""
    profile = load_profile(profile_name)
    install_packages(profile, runner=runner)
    password = secrets.token_urlsafe(24)
    for command in postgres_commands(profile, password) + cache_commands(profile):
        result = runner(command, timeout=300)
        if result.exit_code != 0:
            raise BootstrapError(
                "Local service bootstrap command failed: "
                + (result.stderr or result.stdout or "").strip()[:2000]
            )
    write_env(env_file, build_env(profile, password))
    verify(profile, env_file, runner=runner)
    return {"profile": profile_name, "env_file": str(env_file)}


def verify(profile, env_file, runner=execute):
    """Check that both services answer on localhost; never print credentials."""
    values = read_env(env_file)
    expected = build_env(profile, "placeholder")
    if set(values) != set(expected):
        raise BootstrapError("Service env file is incomplete")
    for command in readiness_commands(profile):
        result = runner(command, timeout=60)
        if result.exit_code != 0:
            raise BootstrapError(
                "Local service readiness check failed: "
                + (result.stderr or result.stdout or "").strip()[:500]
            )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--env-file", required=True)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only verify an already bootstrapped service pair.",
    )
    args = parser.parse_args()

    try:
        profile = load_profile(args.profile)
        if args.check_only:
            verify(profile, args.env_file)
        else:
            bootstrap(args.profile, args.env_file)
    except (BootstrapError, OSError, ValueError) as exc:
        print(f"Service bootstrap failed: {exc}", file=sys.stderr)
        return 1
    print("Temporary local services are ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
