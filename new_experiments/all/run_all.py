# -*- coding: utf-8 -*-
"""Run the paper-aligned HotpotQA experiment flow."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from new_experiments.core import PaperExperimentRunner, RetrievalConfig


def build_config(args: argparse.Namespace) -> RetrievalConfig:
    return RetrievalConfig(
        sample_size=args.sample_size,
        max_workers=args.max_workers,
        run_generation=not args.no_generation,
        run_judge=not args.no_judge,
        use_api_reranker=args.api_reranker,
        require_neo4j=not args.allow_no_neo4j,
        complexity_stratified_sampling=not args.no_complexity_stratified_sampling,
        use_llm_router=args.llm_router,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the complete paper experiment flow")
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--no-generation", action="store_true")
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--api-reranker", action="store_true")
    parser.add_argument("--llm-router", action="store_true")
    parser.add_argument("--allow-no-neo4j", action="store_true")
    parser.add_argument("--no-complexity-stratified-sampling", action="store_true")
    parser.add_argument("--exp", type=str, default="all", help="Only 'all' is supported by the paper flow")
    args = parser.parse_args()

    if args.exp != "all":
        raise SystemExit("The paper-aligned runner executes the full flow only; use --exp all.")

    runner = PaperExperimentRunner(build_config(args))
    result = runner.run_all()
    print(result["run_dir"])
    for name, path in result["tables"].items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
