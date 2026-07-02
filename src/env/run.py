from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Ensure local 'env' package is importable when running this file directly
# (e.g. `python src/env/run.py`) without PYTHONPATH set.
# For `python -m env.run`, users should use: PYTHONPATH=src python -m env.run ...
_this_file = Path(__file__).resolve()
_src_dir = _this_file.parents[1]  # .../src/env/run.py -> .../src
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Guard against the ancient third-party 'env' package (kennethreitz/env 0.1.0)
# which uses Python 2 syntax and shadows this project.
try:
    import env as _env
    # Local package provides submodules like env.core; bad one has env.prefix + no subpackages
    import env.core  # will fail for the PyPI 'env' package
except ImportError as _e:
    _bad = False
    if "env.core" in str(_e):
        _bad = True
    else:
        try:
            if _env is not None and not hasattr(_env, "core") and hasattr(_env, "prefix"):
                _bad = True
        except NameError:
            pass
    if _bad:
        raise RuntimeError(
            "Conflicting 'env' package detected (the PyPI 'env' 0.1.0 by Kenneth Reitz).\n"
            "It is Python 2 only and must be removed:\n"
            "    pip uninstall env\n"
            "This project provides its own 'env' package under src/."
        ) from _e
except RuntimeError:
    raise
except Exception:
    pass

from env.core.config import load_config
from env.core.sampling import sample_sequence
from env.domains.executor import DomainToolExecutor
from env.events.blocker import generate_blocker_replacements_by_task
from env.events.controller import EventController
from env.events.noisy import NoisyToolAugmenter
from env.retriever.semantic import SemanticRetriever
from env.runtime.llm import LLMClient
from env.runtime.prompts import PromptManager
from env.runtime.runner import (
    EnvRunner,
    load_all_baseline_tools,
    load_all_blocker_tools,
    load_all_databases,
    load_all_datatypes,
    load_all_noisy_tools,
    load_noisy_tools_file,
    load_paths_set_catalog,
    load_queries,
)


def _coerce_value(raw: str) -> Any:
    if raw.lower() == "true":
        return True
    if raw.lower() == "false":
        return False
    if raw.lower() in {"null", "none", "~"}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def _parse_set_arg(entry: str) -> tuple[str, Any]:
    if "=" not in entry:
        raise argparse.ArgumentTypeError(f"--set argument must be KEY=VALUE, got: {entry!r}")
    key, _, raw_value = entry.partition("=")
    return key.strip(), _coerce_value(raw_value)


def _setup_logging(level: str, console: bool) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = []
    if console:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        handlers.append(handler)
    logging.basicConfig(level=numeric_level, handlers=handlers, force=True)


def _build_cli_overrides(
    set_entries: list[str],
    resume_validation: str,
    collect: bool,
    sample: int | None,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for entry in set_entries:
        key, value = _parse_set_arg(entry)
        overrides[key] = value

    if resume_validation == "strict":
        overrides.setdefault("strict_resume_config_signature", True)
    elif resume_validation == "ignore-signature":
        overrides.setdefault("unsafe_ignore_config_signature_mismatch", True)

    if collect:
        overrides.setdefault("data_collection", {})
        if isinstance(overrides["data_collection"], dict):
            overrides["data_collection"].setdefault("enabled", True)

    if sample is not None:
        overrides.setdefault("query_sample", {})
        if isinstance(overrides["query_sample"], dict):
            overrides["query_sample"]["size"] = sample

    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a PlanBench-XL evaluation or data collection pass.",
    )
    parser.add_argument(
        "--run_config",
        required=True,
        metavar="PATH",
        help="Path to the run YAML config file.",
    )
    parser.add_argument(
        "--set",
        dest="set_entries",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field using dot-notation KEY=VALUE. Repeatable.",
    )
    parser.add_argument(
        "--collect",
        action="store_true",
        help="Enable data collection mode (sets data_collection.enabled=true).",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Run only N queries (sets query_sample.size=N).",
    )
    parser.add_argument(
        "--resume-validation",
        choices=("strict", "warn-signature", "ignore-signature"),
        default="warn-signature",
        dest="resume_validation",
        help=(
            "strict: abort on config signature mismatch; "
            "warn-signature: log a warning and continue (default); "
            "ignore-signature: silently continue."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cli_overrides = _build_cli_overrides(
        set_entries=args.set_entries,
        resume_validation=args.resume_validation,
        collect=args.collect,
        sample=args.sample,
    )

    cfg = load_config(
        run_config_path=Path(args.run_config),
        cli_overrides=cli_overrides if cli_overrides else None,
    )

    _setup_logging(cfg.logging.level, cfg.logging.console)
    logger = logging.getLogger("run")
    logger.info("Run ID: %s", cfg.run_id)
    logger.info("Output: %s", cfg.output.output_dir)

    data_root = cfg.data.baseline_tools_file.parent.parent

    tool_registry = load_all_baseline_tools(data_root)
    datatype_registry = load_all_datatypes(data_root)
    databases = load_all_databases(data_root)
    _, blocker_tools_by_domain = load_all_blocker_tools(data_root)

    if cfg.noise.mode != "none":
        _, noisy_tools_by_domain = load_noisy_tools_file(cfg.data.noisy_tools_file, domain=cfg.domain)
        noisy_augmenter: NoisyToolAugmenter | None = NoisyToolAugmenter(
            noisy_tools_by_domain=noisy_tools_by_domain,
            mode=cfg.noise.mode,
            max_total_tools=cfg.noise.max_total_tools,
        )
    else:
        noisy_augmenter = None

    queries = load_queries(cfg.data.query_file)
    queries = sample_sequence(queries, cfg.query_sample.size, cfg.query_sample.seed)
    paths_set_catalog = load_paths_set_catalog(cfg.data.paths_set_catalog_file)

    blocker_replacements: dict[str, dict[str, list[Any]]] | None = None
    if cfg.blocker.enable_block:
        blocker_replacements = generate_blocker_replacements_by_task(
            paths_set_catalog=paths_set_catalog,
            baseline_tools_path=cfg.data.baseline_tools_file,
            tasks_path=cfg.data.task_file,
            selection_mode=cfg.blocker.selection_mode,
            block_n_per_task=cfg.blocker.block_n_per_task,
            target_remaining_paths=cfg.blocker.target_remaining_paths,
            target_remaining_ratio=cfg.blocker.target_remaining_ratio,
            remaining_tolerance=cfg.blocker.remaining_tolerance,
            min_remaining_paths=cfg.blocker.min_remaining_paths,
            remaining_path_length_objective=cfg.blocker.remaining_path_length_objective,
            blocking_edge_count_objective=cfg.blocker.blocking_edge_count_objective,
            seed=cfg.blocker.seed,
            noise_mode=cfg.blocker.noise_mode,
            fixed_noise_type=cfg.blocker.fixed_noise_type,
            fixed_noise_types=cfg.blocker.fixed_noise_types,
            multi_noise_count=cfg.blocker.multi_noise_count,
            max_combo_candidates=cfg.blocker.max_combo_candidates,
            max_cover_size=cfg.blocker.max_cover_size,
        )

    event_controller = EventController(
        blocker_tools_by_domain=blocker_tools_by_domain,
        enable_block=cfg.blocker.enable_block,
        blocker_replacements_by_task=blocker_replacements,
    )

    retriever = SemanticRetriever(
        tool_registry=tool_registry,
        datatype_registry=datatype_registry,
        embedding_model=cfg.retriever.embedding_model,
    )

    llm_client = LLMClient(model=cfg.model)
    prompt_manager = PromptManager(prompt_dir=cfg.prompt.prompt_dir)
    tool_executor = DomainToolExecutor(databases=databases)

    runner = EnvRunner(
        config=cfg,
        llm_client=llm_client,
        retriever=retriever,
        event_controller=event_controller,
        noisy_tool_augmenter=noisy_augmenter,
        tool_executor=tool_executor,
        prompt_manager=prompt_manager,
        tool_registry=tool_registry,
    )

    logger.info("Running %d queries...", len(queries))
    results = runner.run(queries=queries, paths_set_catalog=paths_set_catalog)

    completed = sum(1 for r in results if r.get("status") == "completed")
    logger.info("Done. %d/%d completed.", completed, len(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
