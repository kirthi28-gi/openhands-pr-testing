import tarfile
from pathlib import Path, PurePosixPath

archive_path = Path("/workspace/pr-repository.tar.gz")
destination = Path("/workspace").resolve()

with tarfile.open(archive_path, "r:gz") as archive:
    members = archive.getmembers()

    if len(members) > 10000:
        raise RuntimeError("Source archive contains too many entries")

    for member in members:
        member_path = PurePosixPath(member.name)
        target = (destination / member.name).resolve()

        if member_path.is_absolute() or ".." in member_path.parts:
            raise RuntimeError(f"Unsafe archive path: {member.name}")

        if target != destination and destination not in target.parents:
            raise RuntimeError(f"Archive path escapes workspace: {member.name}")

        if not (member.isdir() or member.isfile()):
            raise RuntimeError(f"Unsupported archive entry: {member.name}")

    archive.extractall(destination, members=members, filter="data")

archive_path.unlink()
(destination / "tests/generated").mkdir(parents=True, exist_ok=True)
print("PR source archive safely extracted.")
