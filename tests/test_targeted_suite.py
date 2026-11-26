import importlib.util
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / ".github/scripts"))

spec = importlib.util.spec_from_file_location(
    "repository_profiles", ROOT / ".github/scripts/repository_profiles.py"
)
profiles = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profiles)

import pr_test_harness as harness  # noqa: E402

gt_spec = importlib.util.spec_from_file_location(
    "generate_tests_targeted", ROOT / ".github/scripts/generate_tests.py"
)
generate_tests = importlib.util.module_from_spec(gt_spec)
# generate_tests imports optional OpenHands SDK modules at import time; skip
# this test module when those extras are absent.
try:
    gt_spec.loader.exec_module(generate_tests)
except ImportError:
    generate_tests = None


PROFILE = {
    "targeted_test_paths": {
        "saleor/core/cleaners/": ["saleor/core/cleaners/tests/"],
    },
    "always_test_paths": ["tests/generated"],
    "service_profile": "postgres_valkey_local",
}


def write_config(automation, data):
    path = automation / ".github/config/repository_profiles.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return automation


def make_repo(tmp_path):
    for relative in ("saleor/core/cleaners/tests/test_urls.py", "tests/generated/test_x.py"):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("def test_x(): pass\n", encoding="utf-8")
    return tmp_path


def test_profiles_load_only_from_trusted_automation(tmp_path):
    automation = write_config(tmp_path, {
        "version": 1,
        "profiles": {"saleor/saleor": PROFILE},
    })
    loaded = profiles.load_profiles(automation)
    assert "saleor/saleor" in loaded
    assert profiles.profile_for(loaded, "saleor/saleor")["service_profile"] == "postgres_valkey_local"
    assert profiles.profile_for(loaded, "other/repo") is None


def test_malformed_profiles_fail_closed(tmp_path):
    for content in (
        [],
        {"version": 2, "profiles": {}},
        {"version": 1, "profiles": []},
        {"version": 1, "profiles": {"bad/repo": {"targeted_test_paths": []}}},
        {"version": 1, "profiles": {"bad/repo": {
            "targeted_test_paths": {"src/": ["../../etc/"]}
        }}},
    ):
        automation = write_config(tmp_path, content)
        with pytest.raises(profiles.ProfileError):
            profiles.load_profiles(automation)


def test_targeted_paths_come_from_diff_and_config(tmp_path):
    changed = ["saleor/core/cleaners/urls.py", "README.md"]
    paths = profiles.resolve_test_paths(PROFILE, changed)
    assert paths == ["saleor/core/cleaners/tests", "tests/generated"]
    assert profiles.resolve_test_paths(PROFILE, ["README.md"]) == ["tests/generated"]
    assert profiles.resolve_test_paths(None, changed) is None


def test_trusted_command_builder_rejects_unsafe_or_missing_paths(tmp_path):
    repo = make_repo(tmp_path)
    command, allowed, required = profiles.build_targeted_test_command(
        PROFILE, ["saleor/core/cleaners/urls.py"], repo, "/workspace/.venv/bin/python"
    )
    assert command.startswith("/workspace/.venv/bin/python -m pytest")
    assert "saleor/core/cleaners/tests" in command
    assert "tests/generated" in command
    assert allowed == required == {"saleor/core/cleaners/tests", "tests/generated"}

    # tests/generated exists only in the sandbox and is trusted config.
    without_generated = make_repo(tmp_path / "nogenerated")
    shutil.rmtree(without_generated / "tests/generated")
    built, _, _ = profiles.build_targeted_test_command(
        PROFILE,
        ["saleor/core/cleaners/urls.py"],
        without_generated,
        "/workspace/.venv/bin/python",
    )
    assert "tests/generated" in built

    missing_existing = dict(PROFILE, always_test_paths=["tests/missing"])
    with pytest.raises(profiles.ProfileError, match="does not exist"):
        profiles.build_targeted_test_command(
            missing_existing, ["saleor/core/cleaners/urls.py"], repo, "/usr/bin/python"
        )

    unsafe = dict(PROFILE, always_test_paths=["../../etc/"])
    with pytest.raises(profiles.ProfileError):
        profiles.build_targeted_test_command(
            unsafe, ["saleor/core/cleaners/urls.py"], repo, "/usr/bin/python"
        )


def test_harness_accepts_only_trusted_test_paths(tmp_path):
    repo = make_repo(tmp_path)
    command, allowed, required = profiles.build_targeted_test_command(
        PROFILE, ["saleor/core/cleaners/urls.py"], repo, "/workspace/.venv/bin/python"
    )
    assert harness.validate_test_command(command, repo, allowed, required) == command

    # The same command without a trusted allowlist is a suite selector.
    with pytest.raises(ValueError):
        harness.validate_test_command(command, repo)

    with pytest.raises(ValueError):
        harness.validate_test_command(
            "/workspace/.venv/bin/python -m pytest tests/generated", repo, allowed, required
        )

    for bad in (
        "/workspace/.venv/bin/python -m pytest tests/generated saleor/core/cleaners/tests/ -k x",
        "/workspace/.venv/bin/python -m pytest tests/generated saleor/core/cleaners/tests/ --deselect x",
        "/workspace/.venv/bin/python -m pytest tests/generated saleor/core/cleaners/tests/ --ignore x",
    ):
        with pytest.raises(ValueError):
            harness.validate_test_command(bad, repo, allowed, required)


def test_symlinked_profile_path_is_rejected(tmp_path):
    repo = make_repo(tmp_path)
    outside = tmp_path.parent / "outside-tests"
    outside.mkdir(exist_ok=True)
    link = repo / "tests" / "linked"
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(profiles.ProfileError, match="symlink"):
        profiles.validate_selected_paths(repo, ["tests/linked"])


@pytest.mark.skipif(generate_tests is None, reason="OpenHands SDK extras unavailable")
def test_bootstrap_helper_uses_trusted_script_and_check_only(monkeypatch):
    calls = []

    class Upload:
        success = True
        error = ""

    class Result:
        exit_code = 0
        stdout = "ready"
        stderr = ""

    class Workspace:
        def file_upload(self, source_path, destination_path):
            calls.append(("upload", str(source_path), destination_path))
            return Upload()

        def execute_command(self, command, timeout=300):
            calls.append(("execute", command, timeout))
            return Result()

    automation = ROOT
    env_file = generate_tests.bootstrap_sandbox_services(
        Workspace(), automation, "postgres_valkey_local"
    )

    assert env_file == "/workspace/.openhands-services.env"
    assert calls[0][2] == "/workspace/bootstrap_services.py"
    assert "--profile postgres_valkey_local" in calls[1][1]
    assert "--check-only" in calls[2][1]


def test_service_env_wrapper_exports_trusted_settings():
    command = "/workspace/.venv/bin/python -m pytest tests/generated"
    wrapped = harness.with_service_env(
        command, "/workspace/.openhands-services.env"
    )
    assert wrapped.startswith("/bin/sh -c")
    assert "set -a; . /workspace/.openhands-services.env; set +a;" in wrapped
    assert 'exec "$0" "$@"' in wrapped
    assert command in wrapped
    assert harness.with_service_env(command, None) == command
    with pytest.raises(ValueError):
        harness.with_service_env(command, "../../etc/passwd")


def test_service_env_wrapper_reaches_child_process_and_hides_values(tmp_path):
    env_file = tmp_path / "services.env"
    database_url = "postgres://saleor@127.0.0.1:5432/saleor"
    cache_url = "redis://127.0.0.1:6379/0"
    env_file.write_text(
        f"DATABASE_URL={database_url}\nCACHE_URL={cache_url}\n", encoding="utf-8"
    )
    probe = (
        "import json,os; print(json.dumps({"
        "'DATABASE_URL': os.environ.get('DATABASE_URL'), "
        "'CACHE_URL': os.environ.get('CACHE_URL')}))"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(probe)}"
    wrapped = harness.with_service_env(
        command, str(env_file), env_root=f"{tmp_path}/"
    )

    # The report/loggable command contains the env file path, never its values.
    assert database_url not in wrapped
    assert cache_url not in wrapped

    minimal = "env -i PATH=/usr/bin:/bin HOME=/tmp " + wrapped
    result = subprocess.run(
        minimal, shell=True, capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip()) == {
        "DATABASE_URL": database_url,
        "CACHE_URL": cache_url,
    }


def test_sandbox_test_command_keeps_env_wrapper_inside_minimal_environment():
    wrapped = harness.sandbox_test_command(
        "/workspace/.venv/bin/python -m pytest",
        "/workspace/.openhands-services.env",
    )
    assert wrapped.index("env -i") < wrapped.index("/bin/sh -c")
    assert ". /workspace/.openhands-services.env" in wrapped


def test_saleor_profile_is_present_and_valid():
    loaded = profiles.load_profiles(ROOT)
    saleor = profiles.profile_for(loaded, "saleor/saleor")
    assert saleor["service_profile"] == "postgres_valkey_local"
    assert saleor["targeted_test_paths"]["saleor/core/cleaners"] == [
        "saleor/core/cleaners/tests"
    ]
    assert saleor["generated_test_conftest"] == ".github/config/saleor/generated_conftest.py"
    assert (ROOT / saleor["generated_test_conftest"]).is_file()


@pytest.mark.skipif(generate_tests is None, reason="OpenHands SDK extras unavailable")
def test_generated_test_conftest_is_trusted_and_never_overwritten():
    calls = []

    class Upload:
        success = True
        error = ""

    class Workspace:
        def __init__(self, existing=False):
            self.existing = existing

        def execute_command(self, command, timeout=300):
            from types import SimpleNamespace
            calls.append(command)
            if "test -e" in command:
                return SimpleNamespace(
                    exit_code=0 if self.existing else 1, stdout="", stderr=""
                )
            return SimpleNamespace(exit_code=0, stdout="", stderr="")

        def file_upload(self, source_path, destination_path):
            calls.append((str(source_path), destination_path))
            return Upload()

    assert generate_tests.install_generated_test_conftest(
        Workspace(), ROOT, {"generated_test_conftest": ".github/config/saleor/generated_conftest.py"}
    ) is True
    assert any("tests/generated/conftest.py" in str(item) for item in calls)

    calls.clear()
    with pytest.raises(RuntimeError, match="Untrusted"):
        generate_tests.install_generated_test_conftest(
            Workspace(existing=True),
            ROOT,
            {"generated_test_conftest": ".github/config/saleor/generated_conftest.py"},
        )
    assert not any(isinstance(item, tuple) for item in calls)


@pytest.mark.skipif(generate_tests is None, reason="OpenHands SDK extras unavailable")
def test_bootstrap_helper_fails_closed(monkeypatch):
    class Upload:
        success = True
        error = ""

    class Failed:
        exit_code = 1
        stdout = ""
        stderr = "bootstrap failed"

    class Workspace:
        def file_upload(self, source_path, destination_path):
            return Upload()

        def execute_command(self, command, timeout=300):
            return Failed()

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        generate_tests.bootstrap_sandbox_services(
            Workspace(), ROOT, "postgres_valkey_local"
        )
