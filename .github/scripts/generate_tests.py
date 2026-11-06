
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

            # Environment preparation will be handled by the
            # repository-aware dependency preparation phase.
            # Do not install repository-specific packages here.

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

PR_DIFF.md contains the Git diff for this pull request.
It is reference material only, not executable source code.

ENVIRONMENT PREPARATION — COMPLETE THIS FIRST

Before generating tests, prepare the sandbox environment.

1. Inspect the operating system, available runtimes, tools,
   project configuration, dependency manifests and lockfiles.

2. Identify all relevant programming languages, runtime versions,
   package managers and test frameworks used by this repository.

3. Inspect source imports, build configuration and existing tests
   to identify dependencies not explicitly declared in manifests.

4. Install declared project and development dependencies using
   their appropriate package managers.

5. Detect missing native libraries and system dependencies.
   For compiled extensions, inspect missing shared libraries
   using tools such as ldd where applicable.

6. Identify the actual OS packages providing missing libraries.
   Use available package metadata and package-search tools.
   Do not guess package names or hardcode dependencies for
   a particular repository.

7. Install resolvable dependencies inside the isolated sandbox.
   Use non-interactive package installation and retry transient
   network failures.

8. Verify runtimes, imports, libraries and test tools after
   installation.

9. Do not install unrelated packages or modify application source.
   Do not execute arbitrary installation scripts from the
   repository without inspecting them first.

10. If a dependency cannot be resolved or installed, report the
    exact missing dependency and reason. Do not claim the
    environment is ready.

Only proceed to test generation after environment preparation.

YOUR TASK

1. Inspect the repository structure and identify its application code,
   existing tests, test framework, and dependency configuration.
2. Read PR_DIFF.md as plain text to identify changed files and changes.
3. Never execute, import, or interpret PR_DIFF.md as Python or any
   other programming language.
4. Inspect the actual changed source files in the repository.
5. Understand the intended behavior using the PR description,
   changed code, and relevant existing implementation.
6. Generate meaningful pytest tests for the changed application
   functionality.
7. Cover normal cases, edge cases, invalid inputs, and failure cases
   where relevant.
8. Follow the repository's existing testing conventions.
9. Save newly generated tests under tests/generated/.
10. Use descriptive test names.
11. Do not modify application source code.
12. Do not modify or delete existing tests.
13. Do not overwrite existing generated tests.
14. Do not generate tests that merely inspect this PR-testing
    automation's own workflow or scripts. Only do so if those files
    are themselves the intended application change in this PR.
15. Do not hardcode expected results merely to make tests pass.
16. Do not claim tests passed unless they were actually executed.
17. If the intended behavior is unclear, explain the uncertainty
    instead of inventing expected behavior.

Generate actual pytest test files.

Do not stop after describing the tests. Create the files.
"""

            max_repair_attempts = int(
                os.getenv("MAX_REPAIR_ATTEMPTS", "3")
            )

            try:
                # First, prepare the environment and generate tests.
                conversation.send_message(prompt)
                conversation.run()

                # Run tests and let the same agent diagnose recoverable failures.
                for attempt in range(max_repair_attempts + 1):
                    print(
                        f"\n========== PYTEST RUN "
                        f"{attempt + 1}/{max_repair_attempts + 1} =========="
                    )

                    test_result = workspace.execute_command(
                        "cd /workspace && python -m pytest -q",
                        timeout=900,
                    )

                    print(test_result.stdout or "")

                    if test_result.stderr:
                        print(test_result.stderr)

                    if test_result.exit_code == 0:
                        print("All tests passed.")
                        break

                    if attempt == max_repair_attempts:
                        print("Maximum repair attempts reached.")
                        break

                    failure_output = (
                        (test_result.stdout or "")
                        + "\n"
                        + (test_result.stderr or "")
                    )

                    repair_prompt = f"""
The latest pytest run failed.

Attempt: {attempt + 1} of {max_repair_attempts}

Here is the actual test output:

{failure_output}

Diagnose the failure before making changes.

Rules:
1. Identify whether this is an environment/dependency problem,
   a generated-test problem, an infrastructure problem, or a
   genuine application behavior failure.
2. Repair missing or incorrectly configured dependencies when
   they can be safely resolved.
3. Correct generated tests if their assumptions or assertions
   are wrong. Do not weaken valid assertions just to pass.
4. Do not modify application source code to hide or fix a
   product bug.
5. Do not modify the PR diff or GitHub workflow.
6. Do not claim success without rerunning the tests.
7. If the failure is a genuine application bug or cannot be
   safely repaired, explain why instead of masking it.

Make only the necessary safe repairs, then stop.
The Python runner will execute pytest again.
"""

                    conversation.send_message(repair_prompt)
                    conversation.run()

            finally:
                conversation.close()

            print("\n========== FINAL PYTEST RESULTS ==========")
            print(test_result.stdout or "")

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
