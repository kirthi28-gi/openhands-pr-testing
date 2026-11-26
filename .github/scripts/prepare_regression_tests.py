import argparse
import json
import tarfile
from pathlib import Path, PurePosixPath

EXPECTED_REPOSITORY = "kirthi28-gi/openhands-pr-testing"
METADATA_DIR = Path("metadata/openhands-pr-testing")
TEST_ROOT = Path("tests/generated/openhands-pr-testing")

ALLOWED_SUFFIXES = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    ".java", ".go", ".rb", ".php", ".cs", ".swift",
    ".kt", ".rs", ".feature",
}

MAX_FILES = 1000
MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_TOTAL_SIZE = 50 * 1024 * 1024

REGRESSION_CONFIG = Path(".github/config/regression_namespaces.json")
LEGACY_NAMESPACE = {
    "repository": EXPECTED_REPOSITORY,
    "metadata_dir": METADATA_DIR,
    "test_root": TEST_ROOT,
    "archive_namespace": "openhands-pr-testing",
}


def load_namespaces(config_path=None):
    """Load trusted per-repository regression namespaces."""
    path = Path(config_path) if config_path else (
        Path(__file__).resolve().parents[2] / REGRESSION_CONFIG
    )
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise RuntimeError(f"Could not read regression namespaces: {exc}") from exc
    if not isinstance(data, dict) or data.get("version") != 1:
        raise RuntimeError("Regression namespace config must be a version-1 object")
    repositories = data.get("repositories", {})
    if not isinstance(repositories, dict):
        raise RuntimeError("repositories must be an object")

    namespaces = {}
    for repository, value in repositories.items():
        if not isinstance(repository, str) or repository.count("/") != 1 or not repository.strip():
            raise RuntimeError(f"Invalid repository key: {repository}")
        if not isinstance(value, dict):
            raise RuntimeError(f"Namespace must be an object: {repository}")
        try:
            metadata_dir = safe_relative_path(value.get("metadata_dir"))
            test_root = safe_relative_path(value.get("test_root"))
        except ValueError as exc:
            raise RuntimeError(f"Unsafe namespace path for {repository}: {exc}") from exc
        namespace = {
            "repository": repository,
            "metadata_dir": metadata_dir,
            "test_root": test_root,
            "archive_namespace": value.get("archive_namespace"),
        }
        archive = namespace["archive_namespace"]
        if not isinstance(archive, str) or not archive or "/" in archive or "\\" in archive:
            raise RuntimeError(f"Invalid archive namespace: {repository}")
        namespaces[repository] = namespace
    return namespaces


def namespace_for(source_repository=None):
    """Resolve a trusted namespace; unknown repositories fail closed."""
    if source_repository in (None, ""):
        return dict(LEGACY_NAMESPACE)
    namespaces = load_namespaces()
    if source_repository in namespaces:
        return namespaces[source_repository]
    if source_repository == EXPECTED_REPOSITORY:
        return dict(LEGACY_NAMESPACE)
    raise RuntimeError(
        f"No approved regression namespace for source repository: {source_repository}"
    )


def safe_relative_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Test file path must be a non-empty string")

    path = PurePosixPath(value)

    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError(f"Unsafe test path: {value}")

    return path


def prepare(regression_repo, output, source_repository=None):
    regression_repo = regression_repo.resolve()
    output = output.resolve()
    namespace = namespace_for(source_repository)
    expected_repository = namespace["repository"]

    metadata_root = regression_repo / namespace["metadata_dir"]
    if not metadata_root.is_dir():
        raise RuntimeError(f"Missing metadata directory: {metadata_root}")

    approved_files = {}
    skipped = 0
    total_size = 0

    for metadata_file in sorted(metadata_root.glob("PR-*.json")):
        if metadata_file.is_symlink() or not metadata_file.is_file():
            raise RuntimeError(f"Invalid metadata file: {metadata_file}")

        try:
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not read metadata {metadata_file.name}: {exc}"
            ) from exc

        if (
            metadata.get("validation_status") != "passed"
            or metadata.get("approval_status") != "approved"
        ):
            skipped += 1
            continue

        if metadata.get("source_repository") != expected_repository:
            raise RuntimeError(
                f"Unexpected source repository in {metadata_file.name}"
            )

        pr_number = metadata.get("source_pr_number")
        if type(pr_number) is not int or pr_number < 1:
            raise RuntimeError(f"Invalid PR number in {metadata_file.name}")

        if metadata_file.stem != f"PR-{pr_number}":
            raise RuntimeError(
                f"Metadata filename does not match PR number: {metadata_file.name}"
            )

        files = metadata.get("test_files")
        if not isinstance(files, list) or not files:
            raise RuntimeError(
                f"Approved metadata has no test_files: {metadata_file.name}"
            )

        expected_prefix = namespace["test_root"] / f"PR-{pr_number}"

        for entry in files:
            relative = safe_relative_path(entry)

            if relative.suffix.lower() not in ALLOWED_SUFFIXES:
                raise RuntimeError(f"Unsupported test file type: {entry}")

            if relative.parts[:len(expected_prefix.parts)] != expected_prefix.parts:
                raise RuntimeError(
                    f"Test file is outside its PR directory: {entry}"
                )

            source = regression_repo.joinpath(*relative.parts)

            if source.is_symlink() or not source.is_file():
                raise RuntimeError(f"Missing or unsafe test file: {entry}")

            resolved = source.resolve()
            if not resolved.is_relative_to(regression_repo):
                raise RuntimeError(f"Test file escapes repository: {entry}")

            size = source.stat().st_size
            if size == 0 or size > MAX_FILE_SIZE:
                raise RuntimeError(f"Invalid test file size: {entry}")

            content = source.read_bytes()
            if b"\0" in content:
                raise RuntimeError(f"Binary test file is not allowed: {entry}")

            archive_name = (
                Path("tests/regression")
                / namespace["archive_namespace"]
                / f"PR-{pr_number}"
                / relative.name
            ).as_posix()

            if archive_name in approved_files:
                raise RuntimeError(f"Duplicate regression test: {archive_name}")

            total_size += size
            if total_size > MAX_TOTAL_SIZE:
                raise RuntimeError("Approved regression tests exceed 50 MiB")

            approved_files[archive_name] = content

            if len(approved_files) > MAX_FILES:
                raise RuntimeError("Too many approved regression test files")

    output.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(output, "w:gz") as archive:
        directory = tarfile.TarInfo("tests/regression/")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        archive.addfile(directory)

        for name, content in sorted(approved_files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = 0
            archive.addfile(info, __import__("io").BytesIO(content))

    print(f"Approved regression tests: {len(approved_files)}")
    print(f"Skipped unapproved metadata files: {skipped}")
    print(f"Archive created: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regression-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source-repository",
        default=None,
        help="Repository whose approved tests are being prepared.",
    )
    args = parser.parse_args()

    prepare(args.regression_repo, args.output, args.source_repository)


if __name__ == "__main__":
    main()
