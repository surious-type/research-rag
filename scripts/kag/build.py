import argparse
import time
from pathlib import Path

import yaml

from kag.common.conf import init_env
from kag.builder.main_builder import (
    BuilderMain,
)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
    )

    parser.add_argument(
        "--input",
        required=True,
    )

    args = parser.parse_args()

    config_path = (
        Path(args.config)
        .resolve()
    )

    input_path = (
        Path(args.input)
        .resolve()
    )

    init_env(
        str(config_path)
    )

    with config_path.open(
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(
            file
        )

    builder = BuilderMain(
        config
    )

    started = time.time()

    print(
        "KAG build started",
        flush=True,
    )

    print(
        "Input:",
        input_path,
        flush=True,
    )

    result = builder.invoke(
        str(input_path)
    )

    print(
        "Result:",
        result,
    )

    print(
        "Duration:",
        round(
            time.time()
            - started,
            3,
        ),
        "seconds",
    )


if __name__ == "__main__":
    main()
