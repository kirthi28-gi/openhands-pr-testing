"""Permanent unit tests for the trusted regression-test archive scripts."""

import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".github/scripts"))

from prepare_regression_tests import (  # noqa: E402
    EXPECTED_REPOSITORY,
    METADATA_DIR,
    TEST_ROOT,
    prepare,
    safe_relative_path,
)
from safe_extract_regression_tests import extract  # noqa: E402

APPROVED_CONTENT = "def test_approved(): pass\n"
ARCHIVED_TEST = "tests/regression/openhands-pr-testing/PR-20/test_regression.py"


def _test_relative(pr_number=20, name="test_regression.py"):
    return (TEST_ROOT / f"PR-{pr_number}" / name).as_posix()


def _write_metadata(repo, pr_number=20, files=None, **overrides):
    metadata = {
        "validation_status": "passed",
        "approval_status": "approved",
        "source_repository": EXPECTED_REPOSITORY,
        "source_pr_number": pr_number,
        "test_files": files if files is not None else [_test_relative(pr_number)],
    }
    metadata.update(overrides)
    metadata_dir = repo / METADATA_DIR
    metadata_dir.mkdir(parents=True, exist_ok=True)
    path = metadata_dir / f"PR-{pr_number}.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    return path


def _write_test(repo, relative=None, content=APPROVED_CONTENT):
    relative = relative or _test_relative()
    path = repo / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _tar_dir(name, mode=0o755):
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = mode
    info.size = 0
    return info


def _tar_file(name, data=APPROVED_CONTENT.encode(), mode=0o644):
    info = tarfile.TarInfo(name)
    info.type = tarfile.REGTYPE
    info.mode = mode
    info.size = len(data)
    return info


def _write_archive(path, entries):
    with tarfile.open(path, "w:gz") as archive:
        for info, data in entries:
            archive.addfile(info, io.BytesIO(data) if data is not None else None)


def _archive_entries(root_name="tests/regression/", file_name=ARCHIVED_TEST):
    return [
        (_tar_dir(root_name), None),
        (_tar_file(file_name), APPROVED_CONTENT.encode()),
    ]


# --- prepare_regression_tests.py -------------------------------------------


def test_prepare_packages_only_approved_files_and_normalizes_directory(tmp_path):
    repo = tmp_path / "regression"
    _write_metadata(repo)
    _write_test(repo)
    output = tmp_path / "approved.tar.gz"

    prepare(repo, output)

    with tarfile.open(output, "r:gz") as archive:
        names = archive.getnames()
        assert "tests/regression" in names
        assert "tests/regression/" not in names
        assert ARCHIVED_TEST in names
        content = archive.extractfile(ARCHIVED_TEST).read().decode()
        assert content == APPROVED_CONTENT


def test_prepare_skips_unapproved_metadata(tmp_path):
    repo = tmp_path / "regression"
    _write_metadata(repo, approval_status="pending")
    _write_test(repo)
    output = tmp_path / "approved.tar.gz"

    prepare(repo, output)

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == ["tests/regression"]


@pytest.mark.parametrize(
    "overrides",
    [
        {"validation_status": "failed"},
        {"approval_status": "rejected"},
    ],
)
def test_prepare_skips_invalid_validation_or_approval_status(tmp_path, overrides):
    repo = tmp_path / "regression"
    _write_metadata(repo, **overrides)
    _write_test(repo)
    output = tmp_path / "approved.tar.gz"

    prepare(repo, output)

    with tarfile.open(output, "r:gz") as archive:
        assert archive.getnames() == ["tests/regression"]


def test_prepare_requires_metadata_directory(tmp_path):
    with pytest.raises(RuntimeError, match="Missing metadata directory"):
        prepare(tmp_path / "missing", tmp_path / "approved.tar.gz")


def test_prepare_rejects_malformed_metadata(tmp_path):
    repo = tmp_path / "regression"
    metadata_dir = repo / METADATA_DIR
    metadata_dir.mkdir(parents=True)
    (metadata_dir / "PR-20.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Could not read metadata"):
        prepare(repo, tmp_path / "approved.tar.gz")


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"source_repository": "other/repo"}, "Unexpected source repository"),
        ({"source_pr_number": 21}, "does not match PR number"),
        ({"source_pr_number": 0}, "Invalid PR number"),
        ({"test_files": "not-a-list"}, "no test_files"),
        ({"test_files": []}, "no test_files"),
    ],
)
def test_prepare_rejects_invalid_metadata_fields(tmp_path, overrides, expected):
    repo = tmp_path / "regression"
    _write_metadata(repo, **overrides)
    _write_test(repo)

    with pytest.raises(RuntimeError, match=expected):
        prepare(repo, tmp_path / "approved.tar.gz")


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../evil.py",
        "/etc/passwd",
        "tests/generated/openhands-pr-testing/PR-20/../../evil.py",
        "tests\\generated\\evil.py",
    ],
)
def test_prepare_rejects_path_traversal_in_test_files(tmp_path, unsafe_path):
    repo = tmp_path / "regression"
    _write_metadata(repo, files=[unsafe_path])

    with pytest.raises(ValueError, match="Unsafe test path"):
        prepare(repo, tmp_path / "approved.tar.gz")


@pytest.mark.parametrize(
    ("test_file", "expected"),
    [
        ("tests/generated/openhands-pr-testing/PR-20/test_regression.txt", "Unsupported test file type"),
        ("tests/generated/openhands-pr-testing/PR-21/test_regression.py", "outside its PR directory"),
    ],
)
def test_prepare_rejects_unsupported_or_out_of_scope_paths(tmp_path, test_file, expected):
    repo = tmp_path / "regression"
    _write_metadata(repo, files=[test_file])

    with pytest.raises(RuntimeError, match=expected):
        prepare(repo, tmp_path / "approved.tar.gz")


def test_prepare_rejects_symlinked_metadata(tmp_path):
    repo = tmp_path / "regression"
    real = _write_metadata(repo, pr_number=21)
    metadata_dir = repo / METADATA_DIR
    link = metadata_dir / "PR-20.json"
    link.symlink_to(real)
    (metadata_dir / "PR-21.json").unlink()

    with pytest.raises(RuntimeError, match="Invalid metadata file"):
        prepare(repo, tmp_path / "approved.tar.gz")


def test_prepare_rejects_symlinked_test_file(tmp_path):
    repo = tmp_path / "regression"
    _write_metadata(repo)
    target = _write_test(repo, relative=_test_relative(name="real.py"))
    approved = repo / _test_relative()
    approved.parent.mkdir(parents=True, exist_ok=True)
    approved.symlink_to(target)

    with pytest.raises(RuntimeError, match="Missing or unsafe test file"):
        prepare(repo, tmp_path / "approved.tar.gz")


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("", "Invalid test file size"),
        ("def test_x(): pass\x00\n", "Binary test file is not allowed"),
    ],
)
def test_prepare_rejects_empty_or_binary_tests(tmp_path, content, expected):
    repo = tmp_path / "regression"
    _write_metadata(repo)
    path = repo / _test_relative()
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")

    with pytest.raises(RuntimeError, match=expected):
        prepare(repo, tmp_path / "approved.tar.gz")


def test_prepare_rejects_duplicate_archive_names(tmp_path):
    repo = tmp_path / "regression"
    first = _test_relative(name="one/test_regression.py")
    second = _test_relative(name="two/test_regression.py")
    _write_metadata(repo, files=[first, second])
    _write_test(repo, relative=first)
    _write_test(repo, relative=second)

    with pytest.raises(RuntimeError, match="Duplicate regression test"):
        prepare(repo, tmp_path / "approved.tar.gz")


def test_safe_relative_path_rejects_absolute_and_traversal_paths():
    for value in ("", "../evil.py", "/etc/passwd", "a\\b.py"):
        with pytest.raises(ValueError, match="Unsafe test path|non-empty string"):
            safe_relative_path(value)


# --- safe_extract_regression_tests.py --------------------------------------


@pytest.mark.parametrize("root_name", ["tests/regression/", "tests/regression"])
def test_extract_accepts_normalized_or_slashed_root(tmp_path, root_name):
    archive = tmp_path / "approved.tar.gz"
    _write_archive(archive, _archive_entries(root_name=root_name))

    extract(archive, tmp_path)

    extracted = tmp_path / ARCHIVED_TEST
    assert extracted.is_file()
    assert extracted.read_text(encoding="utf-8") == APPROVED_CONTENT
    assert not archive.exists()


def test_extract_rejects_archive_without_root_directory(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    _write_archive(archive, [(_tar_file(ARCHIVED_TEST), APPROVED_CONTENT.encode())])

    with pytest.raises(RuntimeError, match="no tests/regression directory"):
        extract(archive, tmp_path)


@pytest.mark.parametrize(
    "unsafe_name",
    [
        "../evil.py",
        "/etc/passwd",
        "tests/regression/../../evil.py",
        "tests\\regression\\evil.py",
    ],
)
def test_extract_rejects_path_traversal_members(tmp_path, unsafe_name):
    archive = tmp_path / "approved.tar.gz"
    entries = _archive_entries()
    entries.append((_tar_file(unsafe_name, b"x"), b"x"))
    _write_archive(archive, entries)

    with pytest.raises(RuntimeError, match="Unsafe regression path"):
        extract(archive, tmp_path)


def test_extract_rejects_symlink_member(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    link = tarfile.TarInfo("tests/regression/link.py")
    link.type = tarfile.SYMTYPE
    link.linkname = "../../evil.py"
    link.mode = 0o777
    entries = _archive_entries()
    entries.append((link, None))
    _write_archive(archive, entries)

    with pytest.raises(RuntimeError, match="Unsupported archive entry"):
        extract(archive, tmp_path)


def test_extract_rejects_symlinked_archive_and_destination(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    _write_archive(archive, _archive_entries())
    link = tmp_path / "link.tar.gz"
    link.symlink_to(archive)

    with pytest.raises(RuntimeError, match="missing or unsafe"):
        extract(link, tmp_path / "dest")

    destination = tmp_path / "destination"
    destination.symlink_to(tmp_path)
    with pytest.raises(RuntimeError, match="cannot be a symlink"):
        extract(archive, destination)


def test_extract_prevents_overwriting_existing_regression_tests(tmp_path):
    (tmp_path / "tests/regression").mkdir(parents=True)
    archive = tmp_path / "approved.tar.gz"
    _write_archive(archive, _archive_entries())

    with pytest.raises(RuntimeError, match="Regression test path already exists"):
        extract(archive, tmp_path)


def test_extract_rejects_file_directory_conflicts(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    nested = "tests/regression/openhands-pr-testing/PR-20/test_regression.py/child.py"
    entries = _archive_entries()
    entries.append((_tar_file(nested, b"x"), b"x"))
    _write_archive(archive, entries)

    with pytest.raises(RuntimeError, match="File/directory path conflict"):
        extract(archive, tmp_path)


def test_extract_rejects_duplicate_archive_entries(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    entries = _archive_entries()
    entries.append((_tar_file(ARCHIVED_TEST), APPROVED_CONTENT.encode()))
    _write_archive(archive, entries)

    with pytest.raises(RuntimeError, match="Duplicate archive entry"):
        extract(archive, tmp_path)


def test_extract_rejects_special_permissions_and_empty_files(tmp_path):
    special = tmp_path / "special.tar.gz"
    info = _tar_file(ARCHIVED_TEST)
    info.mode = 0o4755
    _write_archive(special, [(_tar_dir("tests/regression/"), None), (info, APPROVED_CONTENT.encode())])

    with pytest.raises(RuntimeError, match="Special permissions"):
        extract(special, tmp_path)

    empty = tmp_path / "empty.tar.gz"
    empty_info = _tar_file(ARCHIVED_TEST, b"")
    _write_archive(empty, [(_tar_dir("tests/regression/"), None), (empty_info, b"")])

    with pytest.raises(RuntimeError, match="Empty regression test"):
        extract(empty, tmp_path)


def test_extract_rejects_invalid_archive(tmp_path):
    archive = tmp_path / "approved.tar.gz"
    archive.write_bytes(b"not a tar file")

    with pytest.raises(tarfile.ReadError):
        extract(archive, tmp_path)


def test_prepare_rejects_boolean_pr_number(tmp_path):
    repo = tmp_path / "regression"
    _write_metadata(repo, source_pr_number=True)
    _write_test(repo)

    with pytest.raises(RuntimeError, match="Invalid PR number"):
        prepare(repo, tmp_path / "approved.tar.gz")
