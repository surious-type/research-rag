from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from .data import RESULTS_DIR
from .models import FRAMEWORKS, SMOKE_FRAMEWORKS
from .reporting import build_report
from .workflows.check_workflow import (
    check_environment,
    format_checks,
)
from .workflows.run_workflow import (
    execute_run,
    rerun_query_stage,
    rerun_ragas_stage,
    verify_run,
)


def main(argv: list[str] | None = None) -> int:
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
        checks = check_environment()
        print(format_checks(checks))
        return 1 if any(row.status == "FAIL" for row in checks) else 0

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
        for framework in FRAMEWORKS:
            run_id, _ = execute_run(framework, smoke=False)
            print(run_id)
        return 0

    if args.command == "report":
        summary = build_report()
        print(summary)
        return 0

    if args.command == "query":
        run_dir = rerun_query_stage(args.run_id)
        answers_path = run_dir / "query" / "answers.jsonl"
        payload = [json.loads(line) for line in answers_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if args.command == "ragas":
        run_dir = rerun_ragas_stage(args.run_id)
        print((run_dir / "ragas" / "summary.json").read_text(encoding="utf-8"))
        return 0

    if args.command == "verify":
        print(json.dumps(verify_run(args.run_id), ensure_ascii=False, indent=2))
        return 0

    return 1
