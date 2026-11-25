"""Additional behavioral regression coverage for the PR test harness."""

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github" / "scripts"))

import generate_tests
import package_proxy
from safe_extract_regression_tests import extract


def _archive(entries):
    """Build an in-memory tar.gz archive from (name, kind, content) tuples."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, kind, content in entries:
            member = tarfile.TarInfo(name)
            if kind == "directory":
                member.type = tarfile.DIRTYPE
                archive.addfile(member)
            elif kind == "file":
                payload = content.encode()
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = content
                archive.addfile(member)
    buffer.seek(0)
    return buffer


def test_proxy_host_allowlist_accepts_exact_host_and_subdomains_only():
    allowed = ("pypi.org", "files.pythonhosted.org")

    assert package_proxy.allowed_host("pypi.org", allowed)
    assert package_proxy.allowed_host("mirror.files.pythonhosted.org", allowed)
    assert not package_proxy.allowed_host("evilpypi.org", allowed)
    assert not package_proxy.allowed_host("pypi.org.attacker.invalid", allowed)


def test_proxy_public_ip_check_rejects_private_and_metadata_addresses():
    assert package_proxy.is_public_ip("8.8.8.8")
    assert not package_proxy.is_public_ip("127.0.0.1")
    assert not package_proxy.is_public_ip("10.0.0.1")
    assert not package_proxy.is_public_ip("169.254.169.254")
    assert not package_proxy.is_public_ip("::1")


def test_corpus_archive_validator_compiles_python_without_executing_it(tmp_path):
    marker = tmp_path / "executed"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\ndef test_ok(): pass\n"
    path = tmp_path / "corpus.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        root = tarfile.TarInfo("tests/regression")
        root.type = tarfile.DIRTYPE
        archive.addfile(root)
        member = tarfile.TarInfo("tests/regression/test_ok.py")
        payload = source.encode()
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    result = generate_tests.validate_regression_corpus_archive(path)

    assert result["valid"]
    assert result["python_files_compiled"] == 1
    assert not result["executed_against_pr_code"]
    assert not marker.exists()


def test_corpus_archive_validator_rejects_traversal_and_symlinks(tmp_path):
    for kind, name in (("file", "tests/regression/../escape.py"), ("symlink", "tests/regression/link.py")):
        path = tmp_path / f"{kind}.tar.gz"
        with tarfile.open(path, "w:gz") as archive:
            root = tarfile.TarInfo("tests/regression")
            root.type = tarfile.DIRTYPE
            archive.addfile(root)
            member = tarfile.TarInfo(name)
            if kind == "file":
                payload = b"def test_x(): pass\n"
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
            else:
                member.type = tarfile.SYMTYPE
                member.linkname = "../../outside"
                archive.addfile(member)

        result = generate_tests.validate_regression_corpus_archive(path)

        assert not result["valid"]
        assert result["errors"]


def test_safe_regression_extractor_preserves_existing_tests(tmp_path):
    # Archive member names such as tests/regression/test_kept.py are relative
    # to the extraction destination (the workspace root), not to a nested
    # "tests" directory.
    destination = tmp_path
    existing = destination / "tests" / "regression" / "test_kept.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("original\n")
    archive_path = tmp_path / "archive.tar.gz"
    with archive_path.open("wb") as archive_file:
        archive_file.write(_archive([
            ("tests/regression", "directory", ""),
            ("tests/regression/test_kept.py", "file", "replacement\n"),
        ]).getvalue())

    with pytest.raises((FileExistsError, ValueError, RuntimeError)):
        extract(archive_path, destination)

    assert existing.read_text() == "original\n"


def test_safe_regression_extractor_extracts_when_no_existing_test(tmp_path):
    destination = tmp_path
    archive_path = tmp_path / "archive.tar.gz"
    with archive_path.open("wb") as archive_file:
        archive_file.write(_archive([
            ("tests/regression", "directory", ""),
            ("tests/regression/test_kept.py", "file", "replacement\n"),
        ]).getvalue())

    extract(archive_path, destination)

    extracted = destination / "tests" / "regression" / "test_kept.py"
    assert extracted.is_file()
    assert extracted.read_text() == "replacement\n"
    assert not archive_path.exists()


def test_generated_test_validation_command_targets_generated_directory():
    assert generate_tests.GENERATED_TEST_VALIDATION_COMMAND.endswith(
        "-m pytest tests/generated -q"
    )
    assert generate_tests.MAX_GENERATED_TEST_VALIDATION_ATTEMPTS == 3


def test_failure_classifier_uses_inconclusive_for_unsupported_evidence():
    import pr_test_harness

    result = pr_test_harness.classify_failure(
        environment_verification={"ready": True},
        attempt_records=[],
        generated_test_validation=None,
        report={},
        stdout="",
        stderr="",
    )

    assert result["category"] == "inconclusive"
