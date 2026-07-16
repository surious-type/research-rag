from research_bench.workflows.check_workflow import check_environment, format_checks
from research_bench.workflows.run_workflow import execute_run, rerun_query_stage, rerun_ragas_stage, verify_run


def test_workflow_modules_export_public_entrypoints() -> None:
    assert callable(check_environment)
    assert callable(format_checks)
    assert callable(execute_run)
    assert callable(rerun_query_stage)
    assert callable(rerun_ragas_stage)
    assert callable(verify_run)
