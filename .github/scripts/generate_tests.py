
import os
import platform
import shutil
import tarfile
import tempfile
from pathlib import Path

from pydantic import SecretStr

from openhands.sdk import LLM, Conversation
from openhands.tools.preset.default import get_default_agent
from openhands.workspace import DockerWorkspace


def run_checked(workspace, command, timeout=600):
    """Execute a command inside the Docker workspace."""
    result = workspace.execute_command(command, timeout=timeout)

    print(f"\n$ {command}")
    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.exit_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.exit_code}: {command}"
        )

    return result


def create_source_archive(repo_path, diff_path, archive_path):
    """Package PR source code and its diff for sandbox transfer."""

    excluded = {
        ".git",
        ".github",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        "node_modules",
        "generated-tests",
    }

    with tarfile.open(archive_path, "w:gz") as archive:
        for item in repo_path.iterdir():
            if item.name in excluded:
                continue

            archive.add(
                item,
                arcname=item.name,
                recursive=True,
            )

        archive.add(diff_path, arcname="PR_DIFF.md")

    print("PR source archive created.")


def main():
    api_key = os.environ["LLM_API_KEY"]

    repo_path = Path(os.environ["GITHUB_WORKSPACE"]).resolve()
    diff_path = Path(os.environ["PR_DIFF_FILE"]).resolve()

    output_dir = repo_path / "generated-tests"
    output_dir.mkdir(parents=True, exist_ok=True)

    pr_number = os.getenv("PR_NUMBER", "unknown")
    pr_title = os.getenv("PR_TITLE", "")
    pr_body = os.getenv("PR_BODY", "")
    base_sha = os.getenv("PR_BASE_SHA", "")
    head_sha = os.getenv("PR_HEAD_SHA", "")

    llm = LLM(
        usage_id="pr-test-generation",
        model=os.getenv("LLM_MODEL", "gpt-6-luna"),
        api_key=SecretStr(api_key),
        base_url=os.getenv(
            "LLM_BASE_URL",
            "https://api.openai.com/v1",
        ),
        max_output_tokens=3000,
        reasoning_effort="none",
        num_retries=0,
    )

    machine = platform.machine().lower()
    docker_platform = (
        "linux/arm64"
        if "arm" in machine or "aarch64" in machine
        else "linux/amd64"
    )

    server_image = os.getenv(
        "OPENHANDS_SERVER_IMAGE",
        "ghcr.io/openhands/agent-server:latest-python",
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        archive_path = Path(temp_dir) / "pr-repository.tar.gz"

        create_source_archive(
            repo_path,
            diff_path,
            archive_path,
        )

        print("Starting OpenHands Docker sandbox...")

        with DockerWorkspace(
            server_image=server_image,
            platform=docker_platform,
        ) as workspace:

            # Transfer PR source code into the isolated workspace.
            upload = workspace.file_upload(
                source_path=archive_path,
                destination_path="/workspace/pr-repository.tar.gz",
            )

            if not upload.success:
                raise RuntimeError(
                    f"Could not upload PR source: {upload.error}"
                )

            run_checked(
                workspace,
                "mkdir -p /workspace && "
                "tar -xzf /workspace/pr-repository.tar.gz -C /workspace && "
                "rm /workspace/pr-repository.tar.gz && "
                "mkdir -p /workspace/tests/generated",
            )

            # Install project dependencies inside the sandbox.
            install_command = """
cd /workspace
sudo -n apt-get update
sudo -n apt-get install -y libtk8.6
python -m pip install --upgrade pip pytest

if [ -f requirements.txt ]; then
    python -m pip install -r requirements.txt
fi

if [ -f requirements-dev.txt ]; then
    python -m pip install -r requirements-dev.txt
fi
"""
            run_checked(
                workspace,
                install_command,
                timeout=900,
            )

            # Configure the OpenHands agent.
            agent = get_default_agent(
                llm=llm,
                cli_mode=True,
            )

            conversation = Conversation(
                agent=agent,
                workspace=workspace,
            )

            prompt = f"""
You are an experienced Python QA engineer.

Analyze the checked-out pull request repository.

PULL REQUEST INFORMATION

PR number: {pr_number}
Title: {pr_title}

Description:
{pr_body}

Base commit: {base_sha}
Head commit: {head_sha}

The file PR_DIFF.md contains the pull request code diff.

YOUR TASK

1. Inspect the repository structure.
2. Read PR_DIFF.md and identify the changed functionality.
3. Understand the expected behavior from the PR description.
4. Inspect relevant application code and existing tests.
5. Generate meaningful pytest tests for the changed functionality.
6. Cover normal cases, edge cases, negative values and failure cases
   where relevant.
7. Save newly generated tests under tests/generated/.
8. Use descriptive test names.
9. Do not modify application source code.
10. Do not modify or delete existing tests.
11. Do not overwrite existing generated tests.
12. Do not hardcode expected results merely to make tests pass.
13. Do not claim that tests passed unless they were executed.

Generate actual pytest test files.

Do not stop after describing the tests. Create the files.
"""

            try:
                conversation.send_message(prompt)
                conversation.run()
            finally:
                conversation.close()

            # Run the complete existing and newly generated test suite.
            test_result = workspace.execute_command(
                "cd /workspace && python -m pytest -q",
                timeout=900,
            )

            print("\n========== PYTEST RESULTS ==========")
            print(test_result.stdout)

            if test_result.stderr:
                print(test_result.stderr)

            # Package generated tests for retrieval.
            run_checked(
                workspace,
                """
cd /workspace
mkdir -p /workspace/tests/generated
tar -czf /workspace/generated-tests.tar.gz -C /workspace tests/generated
""",
            )

            download = workspace.file_download(
                source_path="/workspace/generated-tests.tar.gz",
                destination_path=str(
                    output_dir / "generated-tests.tar.gz"
                ),
            )

            if not download.success:
                raise RuntimeError(
                    f"Could not download generated tests: {download.error}"
                )

            # Extract generated test files on the GitHub runner.
            downloaded_archive = output_dir / "generated-tests.tar.gz"

            with tarfile.open(downloaded_archive, "r:gz") as archive:
                for member in archive.getmembers():
                    target = (output_dir / member.name).resolve()

                    if not target.is_relative_to(output_dir.resolve()):
                        raise RuntimeError(
                            "Unsafe path found in generated test archive."
                        )

                archive.extractall(output_dir)

            downloaded_archive.unlink(missing_ok=True)

            print("\nGenerated tests retrieved from Docker sandbox.")

            # Publish results in GitHub Actions job summary.
            summary_path = os.getenv("GITHUB_STEP_SUMMARY")

            if summary_path:
                with open(summary_path, "a", encoding="utf-8") as summary:
                    summary.write(
                        f"## OpenHands PR Test Report\n\n"
                        f"- PR: #{pr_number}\n"
                        f"- Base: `{base_sha}`\n"
                        f"- Head: `{head_sha}`\n"
                        f"- Sandbox: `{server_image}`\n\n"
                        f"### Pytest results\n\n"
                        f"```text\n{test_result.stdout}\n```\n"
                    )

            if test_result.exit_code != 0:
                raise RuntimeError(
                    "Pytest reported failures. "
                    "See the GitHub Actions logs for details."
                )

            print("OpenHands test generation and execution completed.")


if __name__ == "__main__":
    main()
