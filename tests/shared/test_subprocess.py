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


def test_run_command_logs_progress_events_when_requested(tmp_path: Path) -> None:
    result = run_command(
        ["bash", "-lc", "printf 'ok'"],
        cwd=tmp_path,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        progress_run_dir=tmp_path,
        progress_stage="build",
        progress_framework="msgraphrag",
        progress_run_id="demo_1",
        process_name="demo process",
    )

    assert result.returncode == 0
    log_text = (tmp_path / "progress.log").read_text(encoding="utf-8")
    assert "event=process_started" in log_text
    assert "event=process_finished" in log_text
