from __future__ import annotations

import argparse
import code
import json
import subprocess
from pathlib import Path

from .data import RESULTS_DIR
from .models import FRAMEWORKS, SMOKE_FRAMEWORKS
from .reporting import build_report
from .workflow import (
    check_environment,
    ensure_source_txt,
    execute_run,
    format_checks,
    verify_run,
)


def main(argv: list[str] | None = None) -> int:
    ensure_source_txt()
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check")
    subparsers.add_parser("test")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("framework", choices=SMOKE_FRAMEWORKS)

    run = subparsers.add_parser("run")
    run.add_argument("framework", choices=FRAMEWORKS)

    subparsers.add_parser("run-all")
    subparsers.add_parser("report")

    query = subparsers.add_parser("query")
    query.add_argument("run_id")

    ragas = subparsers.add_parser("ragas")
    ragas.add_argument("run_id")

    verify = subparsers.add_parser("verify")
    verify.add_argument("run_id")

    args = parser.parse_args(argv)

    if args.command == "check":
        print(format_checks(check_environment()))
        return 0

    if args.command == "test":
        process = subprocess.run(
            [str(Path(".venv/bin/python")), "-m", "pytest", "--cov=research_bench", "--cov-report=term-missing", "--cov-fail-under=80"],
            check=False,
        )
        return process.returncode

    if args.command == "smoke":
        frameworks = FRAMEWORKS if args.framework == "all" else (args.framework,)
        for framework in frameworks:
            run_id, _ = execute_run(framework, smoke=True)
            print(run_id)
        return 0

    if args.command == "run":
        run_id, _ = execute_run(args.framework, smoke=False)
        print(run_id)
        return 0

    if args.command == "run-all":
        print("run-all is implemented but intentionally not executed automatically.")
        return 0

    if args.command == "report":
        summary = build_report()
        print(summary)
        return 0

    if args.command == "query":
        run_dir = RESULTS_DIR / args.run_id
        answers_path = run_dir / "query" / "answers.jsonl"
        payload = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        shell = code.InteractiveConsole(locals={"answers": payload, "run_id": args.run_id})
        shell.interact("answers is available; inspect the loaded query results.")
        return 0

    if args.command == "ragas":
        print((RESULTS_DIR / args.run_id / "ragas" / "summary.json").read_text(encoding="utf-8"))
        return 0

    if args.command == "verify":
        print(json.dumps(verify_run(args.run_id), ensure_ascii=False, indent=2))
        return 0

    return 1
