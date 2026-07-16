from pathlib import Path

from research_bench.shared.subprocess import run_command


def test_run_command_writes_stdout_and_stderr_files(tmp_path: Path) -> None:
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"

    result = run_command(
        ["bash", "-lc", "printf 'out'; printf 'err' >&2"],
        cwd=tmp_path,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    assert result.returncode == 0
    assert stdout_path.read_text(encoding="utf-8") == "out"
    assert stderr_path.read_text(encoding="utf-8").rstrip().endswith("err")
