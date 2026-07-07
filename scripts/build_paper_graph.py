# -*- coding: utf-8 -*-
"""Build the paper structural + semantic graph in Neo4j."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage.graph_store.neo4j_store import Neo4jGraphStore
from src.storage.graph_store.paper_graph_builder import load_paper_graph, select_paper_graph_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Build paper-aligned Neo4j graph")
    parser.add_argument(
        "--documents-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "hotpotqa" / "valid_title_sentence.json",
    )
    parser.add_argument(
        "--test-data-path",
        type=Path,
        default=PROJECT_ROOT / "data" / "hotpotqa" / "validation-00000-of-00001.parquet",
    )
    parser.add_argument("--sample-size", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    with open(args.documents_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.sample_size:
        samples = pd.read_parquet(args.test_data_path)
        if args.sample_size < len(samples):
            samples = samples.sample(n=args.sample_size, random_state=args.random_seed)
        records = select_paper_graph_records(records, samples=samples.to_dict(orient="records"))
        print(f"Selected {len(records)} title records for {len(samples)} sampled questions")

    graph_store = Neo4jGraphStore()
    load_paper_graph(graph_store, records, batch_size=args.batch_size, clear=not args.no_clear)

    counts = graph_store.query(
        """
        MATCH (n)
        RETURN labels(n)[0] AS label, count(n) AS count
        ORDER BY label
        """
    )
    rel_counts = graph_store.query(
        """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(r) AS count
        ORDER BY type
        """
    )
    print("Node counts:")
    for row in counts:
        print(f"{row['label']}: {row['count']}")
    print("Relationship counts:")
    for row in rel_counts:
        print(f"{row['type']}: {row['count']}")
    graph_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
