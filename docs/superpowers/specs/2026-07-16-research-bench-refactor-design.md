# Research Bench Architecture Refactor Design

**Date:** 2026-07-16

## Goal

Провести senior-level рефакторинг проекта `/home/surious-type/projects/research-rag`, чтобы кодовая база стала читаемой, масштабируемой и удобной для сопровождения без переписывания workflow с нуля.

Ключевые ограничения:

- не менять публичный CLI;
- не менять контракт `experiment.py`;
- не менять поведение benchmark pipeline;
- не менять upstream Microsoft GraphRAG, LightRAG и `frameworks/kag`;
- не ломать существующие benchmark artifacts и выходные директории;
- добавить явное логирование стадий и процессов.

## Current Problems

По текущему состоянию проекта основные архитектурные проблемы такие:

- `research_bench/frameworks.py` выполняет слишком много ролей и стал god-file;
- framework-specific логика смешана с общей orchestration и диагностикой;
- `research_bench/workflow.py` содержит orchestration, артефакты и частично verification concerns;
- logging и progress visibility недостаточно явны для долгих запусков;
- структура тестов недостаточно отражает реальные архитектурные границы;
- новому разработчику неочевидно, с какого файла начинать чтение проекта и в каком порядке изучать модули.

## Target Architecture

Проект должен читаться сверху вниз:

`experiment.py`
-> `research_bench/cli/commands.py`
-> `research_bench/workflows/<command>_workflow.py`
-> `research_bench/workflows/stages.py`
-> `research_bench/adapters/registry.py`
-> `research_bench/adapters/<framework>.py`
-> `research_bench/diagnostics|reporting|verification|shared`

Целевая структура:

```text
research_bench/
  __init__.py

  cli/
    __init__.py
    commands.py
    parsing.py

  workflows/
    __init__.py
    run_workflow.py
    check_workflow.py
    test_workflow.py
    smoke_workflow.py
    stages.py

  adapters/
    __init__.py
    base.py
    registry.py
    msgraphrag.py
    lightrag.py
    kag.py

  diagnostics/
    __init__.py
    logging.py
    traces.py
    artifacts.py
    probes.py

  reporting/
    __init__.py
    reports.py
    ragas.py
    summaries.py

  verification/
    __init__.py
    checks.py
    schema.py
    smoke.py

  shared/
    __init__.py
    io.py
    paths.py
    subprocess.py
    text.py
    models.py

  config/
    __init__.py
    settings.py
    constants.py
```

## Layer Responsibilities

### CLI

`cli/` только принимает пользовательскую команду и переводит её в вызов соответствующего workflow. CLI не знает деталей конкретных framework implementations.

### Workflows

`workflows/` управляет последовательностью стадий benchmark lifecycle:

- `prepare`
- `build`
- `query`
- `ragas`
- `report`
- `verify`
- `cleanup`, если применимо

Workflow не должен содержать framework-specific implementation details и низкоуровневый subprocess/file handling.

### Adapters

`adapters/` содержит только framework-specific поведение:

- build;
- query;
- environment preparation;
- framework-specific diagnostics;
- framework-specific data normalization, если она не относится к общему reporting contract.

### Diagnostics

`diagnostics/` отвечает за наблюдаемость:

- stage progress logging;
- subprocess lifecycle logging;
- trace artifacts;
- probe results;
- artifact registration.

### Reporting

`reporting/` строит итоговые summary и RAGAS-related outputs, но не управляет выполнением стадий.

### Verification

`verification/` хранит PASS/WARN/FAIL rules и post-run validation logic.

### Shared

`shared/` хранит переиспользуемую инфраструктуру и простые данные общего назначения. Этот слой не должен стать новой свалкой из разнородных helper functions.

## Dependency Rules

Разрешённые зависимости:

- `cli` -> `workflows`
- `workflows` -> `adapters`, `reporting`, `verification`, `diagnostics`, `shared`
- `adapters` -> `shared`, `diagnostics`
- `reporting` -> `shared`
- `verification` -> `shared`, при необходимости `reporting`
- `diagnostics` -> `shared`
- `config` -> доступен всем как слой настроек и констант

Нежелательные и запрещённые зависимости:

- `adapters` не импортируют `workflows`
- `adapters` не знают о других adapter modules напрямую без явной необходимости
- `reporting` не управляет workflow execution
- `diagnostics` не принимает бизнес-решения benchmark pipeline
- `shared` не импортирует high-level layers

## File Migration Plan

### `experiment.py`

Остаётся текущим entrypoint и сохраняет CLI contract. Внутренние импорты переводятся на новую CLI/workflow структуру.

### `research_bench/frameworks.py`

Содержимое переносится так:

- базовые adapter contracts -> `research_bench/adapters/base.py`
- adapter registry -> `research_bench/adapters/registry.py`
- `GraphRAGAdapter` -> `research_bench/adapters/msgraphrag.py`
- `LightRAGAdapter` -> `research_bench/adapters/lightrag.py`
- `KAGAdapter` и KAG-specific helper logic -> `research_bench/adapters/kag.py`

### `research_bench/workflow.py`

Содержимое переносится так:

- основной benchmark run -> `research_bench/workflows/run_workflow.py`
- `check` flow -> `research_bench/workflows/check_workflow.py`
- `test` flow -> `research_bench/workflows/test_workflow.py`
- `smoke` flow -> `research_bench/workflows/smoke_workflow.py`
- переиспользуемые stage helpers -> `research_bench/workflows/stages.py`

### `research_bench/utils.py`

Безликий `utils.py` раскладывается по ответственности:

- subprocess helpers -> `research_bench/shared/subprocess.py`
- file I/O helpers -> `research_bench/shared/io.py`
- path helpers -> `research_bench/shared/paths.py`
- text helpers -> `research_bench/shared/text.py`

### `research_bench/models.py`

Переносится в `research_bench/shared/models.py`, если там находятся общие dataclass/model structures.

### Reporting and Verification

Итоговая логика переносится в:

- `research_bench/reporting/reports.py`
- `research_bench/reporting/ragas.py`
- `research_bench/reporting/summaries.py`
- `research_bench/verification/checks.py`
- `research_bench/verification/schema.py`
- `research_bench/verification/smoke.py`

## Logging and Observability Design

Для каждого run добавляются два верхнеуровневых лога:

- `progress.log` для чтения человеком;
- `progress.jsonl` для машинно-читаемых событий.

События должны покрывать:

- `run_started`
- `stage_started`
- `stage_completed`
- `stage_failed`
- `subprocess_started`
- `subprocess_completed`
- `subprocess_failed`
- `artifact_written`

Минимальные поля события:

- `timestamp`
- `run_id`
- `framework`
- `stage`
- `event`
- `status`
- `message`
- `artifact_path`, если применимо

Границы ответственности:

- workflow layer логирует start/end/fail стадий;
- adapter layer логирует framework-specific subprocess details и локальные diagnostics;
- большие payload сохраняются в artifacts, а не в основном progress log;
- секреты из environment variables не логируются.

Пользователь должен иметь возможность без остановки long-running benchmark открыть:

- `results/<run_id>/progress.log`
- `results/<run_id>/build/stdout.log`
- `results/<run_id>/build/stderr.log`

и сразу увидеть текущую стадию и активные subprocesses.

## Test Strategy

Структура тестов должна отражать новую архитектуру:

```text
tests/
  adapters/
  workflows/
  diagnostics/
  reporting/
  verification/
  shared/
```

Принципы:

- adapter behavior tests живут в `tests/adapters/`
- orchestration tests живут в `tests/workflows/`
- verification rules живут в `tests/verification/`
- logging/progress/artifact tests живут в `tests/diagnostics/`
- subprocess/path/io helper tests живут в `tests/shared/`

## Safe Execution Order

Рефакторинг выполняется безопасными волнами:

1. Зафиксировать текущую зелёную базу:
   - `python experiment.py check`
   - `python experiment.py test`
   - `pytest -q`
2. Вынести `shared/` и `diagnostics/`
3. Вынести `adapters/` и перевести framework imports
4. Разрезать `workflow.py` на `workflows/`
5. Вынести `reporting/` и `verification/`
6. Перестроить тесты зеркально новой структуре
7. Удалить мёртвые импорты и технические переходные остатки
8. Повторно прогнать весь validation suite

## Success Criteria

Рефакторинг считается успешным, если:

- CLI и `experiment.py` contract не изменились;
- benchmark behavior не изменился;
- `research_bench/frameworks.py` перестал быть главным центром логики;
- структура проекта читается сверху вниз;
- у каждого слоя одна явная зона ответственности;
- progress logging делает long-running execution наблюдаемым;
- тестовая структура отражает новую архитектуру;
- новый разработчик может понять рекомендуемый порядок чтения проекта менее чем за час.

## Recommended Reading Order After Refactor

1. `README.md`
2. `docs/project_map.md`
3. `docs/architecture.md`
4. `docs/reading_order.md`
5. `experiment.py`
6. `research_bench/cli/commands.py`
7. `research_bench/workflows/run_workflow.py`
8. `research_bench/workflows/stages.py`
9. `research_bench/adapters/registry.py`
10. конкретный adapter module
11. `research_bench/reporting/*`
12. `research_bench/verification/*`
13. `tests/*`
