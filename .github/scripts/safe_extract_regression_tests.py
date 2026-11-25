import argparse
import tarfile
from pathlib import Path, PurePosixPath


MAX_ENTRIES = 1000
MAX_TOTAL_SIZE = 100 * 1024 * 1024


def extract(archive_path, destination):
    archive_path = Path(archive_path).absolute()
    destination = Path(destination).absolute()

    if archive_path.is_symlink() or not archive_path.is_file():
        raise RuntimeError("Approved regression archive is missing or unsafe.")

    if destination.is_symlink():
        raise RuntimeError("Extraction destination cannot be a symlink.")

    archive_path = archive_path.resolve()
    destination = destination.resolve()

    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()

        if len(members) > MAX_ENTRIES:
            raise RuntimeError("Too many regression archive entries.")

        total_size = 0
        root_found = False
        seen = set()
        validated = []

        for member in members:
            name = member.name

            if not name or "\\" in name or name.startswith("/"):
                raise RuntimeError(f"Unsafe regression path: {name}")

            normalized = name[:-1] if name.endswith("/") else name
            parts = normalized.split("/")

            if (
                any(part in ("", ".", "..") for part in parts)
                or parts[:2] != ["tests", "regression"]
            ):
                raise RuntimeError(f"Unsafe regression path: {name}")

            path = PurePosixPath(*parts)

            if not (member.isdir() or member.isfile()):
                raise RuntimeError(f"Unsupported archive entry: {name}")

            if member.mode & 0o7000:
                raise RuntimeError(f"Special permissions are not allowed: {name}")

            if name in seen:
                raise RuntimeError(f"Duplicate archive entry: {name}")
            seen.add(name)

            if path.parts == ("tests", "regression"):
                if not member.isdir():
                    raise RuntimeError("Regression root must be a directory.")
                root_found = True

            if member.isdir() and member.size != 0:
                raise RuntimeError(f"Invalid directory entry: {name}")

            if member.isfile():
                if member.size <= 0:
                    raise RuntimeError(f"Empty regression test: {name}")

                total_size += member.size
                if total_size > MAX_TOTAL_SIZE:
                    raise RuntimeError("Regression archive exceeds 100 MiB.")

            target = destination.joinpath(*path.parts)

            if not target.resolve().is_relative_to(destination):
                raise RuntimeError(f"Regression path escapes destination: {name}")

            if target.exists() or target.is_symlink():
                raise RuntimeError(f"Regression test path already exists: {name}")

            validated.append((member, path))

        if not root_found:
            raise RuntimeError(
                "Regression archive has no tests/regression directory."
            )

        # Reject file/directory path conflicts before writing anything.
        file_paths = {
            path for member, path in validated if member.isfile()
        }

        for file_path in file_paths:
            if any(
                file_path in other.parents
                for _, other in validated
                if other != file_path
            ):
                raise RuntimeError(
                    f"File/directory path conflict: {file_path}"
                )

        destination.mkdir(parents=True, exist_ok=True)

        for member, path in validated:
            target = destination.joinpath(*path.parts)

            if not target.resolve().is_relative_to(destination):
                raise RuntimeError(
                    f"Regression path escapes destination: {member.name}"
                )

            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue

            target.parent.mkdir(parents=True, exist_ok=True)

            # Recheck after parent creation to guard against symlink traversal.
            if (
                not target.resolve().is_relative_to(destination)
                or target.exists()
                or target.is_symlink()
            ):
                raise RuntimeError(
                    f"Unsafe or existing regression path: {member.name}"
                )

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(
                    f"Could not read archive entry: {member.name}"
                )

            with source, target.open("xb") as output:
                remaining = member.size

                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise RuntimeError(
                            f"Truncated regression file: {member.name}"
                        )

                    output.write(chunk)
                    remaining -= len(chunk)

    archive_path.unlink()
    print("Approved regression tests safely extracted.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("/workspace/approved-regression-tests.tar.gz"),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/workspace"),
    )
    args = parser.parse_args()

    extract(args.archive, args.destination)

if __name__ == "__main__":
    main()
