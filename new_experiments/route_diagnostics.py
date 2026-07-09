# -*- coding: utf-8 -*-
"""Route and graph diagnostics for the paper HotpotQA flow.

This module intentionally runs retrieval only. It builds a route-balanced
diagnostic set and evaluates counterfactual routes against the same questions.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

from new_experiments.core import (
    EvidenceUnit,
    PaperExperimentRunner,
    RetrievalConfig,
    compute_paper_table_metrics,
    compute_support_fact_metrics,
    context_len,
    estimate_tokens,
    unique_by_title,
)


ROUTES = ("fine_grained", "local_parent", "graph_expansion")
COUNTERFACTUAL_METHODS = ("fine_grained", "local_parent", "graph_expansion", "proposed")


def choose_best_route(metric_rows: Sequence[Dict[str, Any]]) -> str:
    """Choose the best route by recall, then token cost, then expansion cost."""
    route_rows = [row for row in metric_rows if row.get("method") in ROUTES]
    if not route_rows:
        return ""
    best = sorted(
        route_rows,
        key=lambda row: (
            -float(row.get("recall") or 0.0),
            float(row.get("avg_len") or 0.0),
            float(row.get("expanded_nodes") or 0.0),
            ROUTES.index(str(row.get("method"))),
        ),
    )[0]
    return str(best["method"])


def summarize_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("original_route", "")), str(row.get("method", "")))].append(row)

    summaries: List[Dict[str, Any]] = []
    for (original_route, method), rows in sorted(grouped.items()):
        n = len(rows)
        if not n:
            continue
        summaries.append(
            {
                "original_route": original_route,
                "method": method,
                "n": n,
                "recall": round(sum(float(r.get("recall") or 0.0) for r in rows) / n, 4),
                "precision": round(sum(float(r.get("precision") or 0.0) for r in rows) / n, 4),
                "avg_len": round(sum(float(r.get("avg_len") or 0.0) for r in rows) / n, 2),
                "expanded_nodes": round(sum(float(r.get("expanded_nodes") or 0.0) for r in rows) / n, 2),
                "time_ms": round(sum(float(r.get("time_ms") or 0.0) for r in rows) / n, 2),
            }
        )
    return summaries


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def restore_balanced_samples(source_data: pd.DataFrame, balanced_rows: pd.DataFrame) -> pd.DataFrame:
    """Restore balanced CSV IDs to the original dataset rows.

    CSV round-tripping turns nested HotpotQA fields such as supporting_facts
    into strings. Counterfactual scoring must use the original parquet objects,
    while keeping the diagnostic route metadata recorded during sampling.
    """
    by_id = {str(row["id"]): row.to_dict() for _, row in source_data.iterrows()}
    restored: List[Dict[str, Any]] = []
    diagnostic_cols = [col for col in balanced_rows.columns if col.startswith("_")]

    for _, csv_row in balanced_rows.iterrows():
        sample_id = str(csv_row.get("id"))
        if sample_id not in by_id:
            raise KeyError(f"Balanced sample id not found in source data: {sample_id}")
        row = dict(by_id[sample_id])
        for col in diagnostic_cols:
            row[col] = csv_row.get(col)
        restored.append(row)

    return pd.DataFrame(restored).reset_index(drop=True)


class RouteDiagnostics:
    def __init__(self, runner: PaperExperimentRunner):
        self.runner = runner

    def build_route_balanced_samples(
        self,
        per_route: int,
        max_scan: int,
        random_seed: int,
    ) -> pd.DataFrame:
        assert self.runner.test_data is not None
        data = self.runner.test_data.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
        counts = Counter()
        selected: List[Dict[str, Any]] = []

        for scanned, (_, sample) in enumerate(tqdm(data.iterrows(), total=min(max_scan, len(data)), desc="route scan"), start=1):
            if scanned > max_scan or all(counts[route] >= per_route for route in ROUTES):
                break
            question = str(sample.get("question", ""))
            reranked = self.runner.initial_reranked_candidates(question)
            route, detail = self.runner.choose_route(question, reranked)
            if route not in ROUTES or counts[route] >= per_route:
                continue
            row = sample.to_dict()
            row["_diagnostic_route"] = route
            row["_route_detail"] = json.dumps(detail, ensure_ascii=False)
            row["_scan_order"] = scanned
            selected.append(row)
            counts[route] += 1

        missing = {route: per_route - counts[route] for route in ROUTES if counts[route] < per_route}
        if missing:
            raise RuntimeError(f"Could not build route-balanced set; missing={missing}, scanned={sum(counts.values())}/{max_scan}")
        return pd.DataFrame(selected).reset_index(drop=True)

    def _fine_units(self, reranked: Sequence[EvidenceUnit]) -> Tuple[List[EvidenceUnit], int]:
        return unique_by_title(reranked)[: self.runner.config.k3], 0

    def _parent_units(self, query: str, reranked: Sequence[EvidenceUnit]) -> Tuple[List[EvidenceUnit], int]:
        return self.runner.select_with_budget(query, self.runner.parent_context_pool(reranked)), 0

    def _graph_units(
        self,
        query: str,
        reranked: Sequence[EvidenceUnit],
        enable_summary: bool = True,
        forced_hops: Optional[int] = None,
    ) -> Tuple[List[EvidenceUnit], int]:
        route, detail = self.runner.choose_adaptive_route(query, reranked)
        parent_units = [
            unit
            for unit in self.runner.parent_map(reranked)
            if unit.metadata.get("trigger_unit_ids")
        ]
        hops = forced_hops or self.runner._dynamic_hops({**detail, "route": route})
        expanded = self.runner.graph_expand(
            self.runner.graph_seed_titles(reranked),
            hops=hops,
            limit_per_seed=self.runner.config.graph_expansion_limit_per_seed,
            query=query,
            seed_units=reranked,
        )
        summaries: List[EvidenceUnit] = []
        current_tokens = sum(estimate_tokens(u.content) for u in parent_units + expanded)
        if enable_summary and self.runner.config.context_budget * 0.5 - current_tokens > 0:
            summary_seed_units = parent_units or list(reranked)
            summaries = self.runner.add_summary_evidence([u.title for u in summary_seed_units], query)
        return self.runner.select_with_budget(query, list(reranked) + parent_units + expanded + summaries), len(expanded)

    def route_result(
        self,
        query: str,
        route: str,
        reranked: Optional[Sequence[EvidenceUnit]] = None,
    ) -> Tuple[List[EvidenceUnit], Dict[str, Any]]:
        start = time.perf_counter()
        if reranked is None:
            reranked = self.runner.initial_reranked_candidates(query)

        if route == "fine_grained":
            units, expanded = self._fine_units(reranked)
        elif route == "local_parent":
            units, expanded = self._parent_units(query, reranked)
        elif route == "graph_expansion":
            units, expanded = self._graph_units(query, reranked)
        else:
            method_result = self.runner.retrieve_adaptive(query)
            return method_result.units, method_result.stats

        stats = {
            "time_ms": (time.perf_counter() - start) * 1000,
            "avg_len": context_len(units),
            "expanded_nodes": expanded,
            "route": route,
        }
        return units, stats

    def counterfactual_records(self, samples: pd.DataFrame) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        records: List[Dict[str, Any]] = []
        decisions: List[Dict[str, Any]] = []
        assert self.runner.test_data is not None

        for idx, sample in tqdm(list(samples.iterrows()), desc="counterfactual"):
            question = str(sample.get("question", ""))
            relevant_facts = self.runner.get_relevant_facts(sample)
            reranked = self.runner.initial_reranked_candidates(question)
            original_route, route_detail = self.runner.choose_route(question, reranked)
            sample_records: List[Dict[str, Any]] = []

            for method in COUNTERFACTUAL_METHODS:
                if method == "proposed":
                    units, stats = self.route_result(question, original_route, reranked)
                    actual_method = "proposed"
                else:
                    units, stats = self.route_result(question, method, reranked)
                    actual_method = method

                metrics = compute_support_fact_metrics(
                    retrieved_units=units,
                    relevant_facts=relevant_facts,
                    avg_context_len=stats.get("avg_len", context_len(units)),
                    latency_ms=stats.get("time_ms", 0.0),
                    expanded_nodes=stats.get("expanded_nodes", 0),
                )
                paper_metrics = compute_paper_table_metrics(metrics, metrics)
                row = {
                    "id": sample.get("id"),
                    "question": question,
                    "type": sample.get("type"),
                    "level": sample.get("level"),
                    "original_route": original_route,
                    "method": actual_method,
                    "selected_route": stats.get("route", method),
                    "recall": paper_metrics.recall,
                    "precision": paper_metrics.precision,
                    "mrr": paper_metrics.mrr,
                    "ndcg": paper_metrics.ndcg,
                    "map": paper_metrics.map_score,
                    "avg_len": metrics.avg_len,
                    "expanded_nodes": metrics.expanded_nodes,
                    "time_ms": metrics.time_ms,
                    "retrieved_titles": json.dumps([u.title for u in units], ensure_ascii=False),
                    "route_detail": json.dumps(route_detail, ensure_ascii=False),
                }
                sample_records.append(row)
                records.append(row)

            best_route = choose_best_route(sample_records)
            current = next(row for row in sample_records if row["method"] == "proposed")
            best = next(row for row in sample_records if row["method"] == best_route)
            graph = next(row for row in sample_records if row["method"] == "graph_expansion")
            non_graph_best = max(
                [row for row in sample_records if row["method"] in {"fine_grained", "local_parent"}],
                key=lambda row: (row["recall"], -row["avg_len"]),
            )
            decisions.append(
                {
                    "id": sample.get("id"),
                    "question": question,
                    "original_route": original_route,
                    "best_route": best_route,
                    "current_matches_best": original_route == best_route,
                    "current_recall": current["recall"],
                    "best_recall": best["recall"],
                    "current_avg_len": current["avg_len"],
                    "best_avg_len": best["avg_len"],
                    "graph_recall_gain_vs_best_non_graph": round(float(graph["recall"]) - float(non_graph_best["recall"]), 4),
                    "graph_token_delta_vs_best_non_graph": round(float(graph["avg_len"]) - float(non_graph_best["avg_len"]), 2),
                }
            )

        return records, decisions


def main() -> int:
    parser = argparse.ArgumentParser(description="Run route-balanced counterfactual diagnostics")
    parser.add_argument("--per-route", type=int, default=100)
    parser.add_argument("--max-scan", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("new_experiments/results/route_diagnostics"))
    parser.add_argument("--balanced-csv", type=Path, default=None)
    parser.add_argument("--mode", choices=["build", "counterfactual", "all"], default="all")
    parser.add_argument("--llm-router", action="store_true")
    args = parser.parse_args()

    cfg = RetrievalConfig(
        sample_size=0,
        random_seed=args.seed,
        run_generation=False,
        run_judge=False,
        require_neo4j=True,
        complexity_stratified_sampling=False,
        use_llm_router=args.llm_router,
    )
    runner = PaperExperimentRunner(cfg)
    runner.load_resources()
    diagnostics = RouteDiagnostics(runner)

    run_dir = args.output_dir / datetime.now().strftime("route_diag_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.balanced_csv:
        assert runner.test_data is not None
        balanced = restore_balanced_samples(runner.test_data, pd.read_csv(args.balanced_csv))
    else:
        balanced = diagnostics.build_route_balanced_samples(args.per_route, args.max_scan, args.seed)
        balanced.to_csv(run_dir / "route_balanced_samples.csv", index=False, encoding="utf-8")

    if args.mode == "build":
        print(run_dir)
        return 0

    records, decisions = diagnostics.counterfactual_records(balanced)
    write_csv(run_dir / "counterfactual_records.csv", records)
    write_csv(run_dir / "counterfactual_summary.csv", summarize_records(records))
    write_csv(run_dir / "route_decisions.csv", decisions)

    decision_summary = {
        "n": len(decisions),
        "route_counts": dict(Counter(row["original_route"] for row in decisions)),
        "best_route_counts": dict(Counter(row["best_route"] for row in decisions)),
        "current_matches_best": sum(1 for row in decisions if row["current_matches_best"]),
        "graph_positive_gain": sum(1 for row in decisions if float(row["graph_recall_gain_vs_best_non_graph"]) > 0),
        "graph_nonpositive_gain": sum(1 for row in decisions if float(row["graph_recall_gain_vs_best_non_graph"]) <= 0),
    }
    (run_dir / "decision_summary.json").write_text(json.dumps(decision_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(run_dir)
    print(json.dumps(decision_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
