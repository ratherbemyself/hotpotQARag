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
from new_experiments.core import QueryComplexityScorer, sample_hotpotqa_by_complexity


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
    parser.add_argument("--complexity-threshold", type=float, default=0.80)
    parser.add_argument("--no-complexity-stratified-sampling", action="store_true")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--with-embeddings", action="store_true", help="Store local text embeddings on Section/Paragraph/Sentence nodes")
    parser.add_argument("--embedding-device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--embedding-batch-size", type=int, default=1024)
    parser.add_argument("--no-clear", action="store_true")
    args = parser.parse_args()

    with open(args.documents_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    if args.sample_size:
        samples = pd.read_parquet(args.test_data_path)
        if args.sample_size < len(samples):
            if args.no_complexity_stratified_sampling:
                samples = samples.sample(n=args.sample_size, random_state=args.random_seed)
            else:
                samples = sample_hotpotqa_by_complexity(
                    samples,
                    sample_size=args.sample_size,
                    scorer=QueryComplexityScorer(),
                    random_seed=args.random_seed,
                    complexity_threshold=args.complexity_threshold,
                )
        records = select_paper_graph_records(records, samples=samples.to_dict(orient="records"))
        print(f"Selected {len(records)} title records for {len(samples)} sampled questions")

    graph_store = Neo4jGraphStore()
    embedding_client = None
    if args.with_embeddings:
        from src.llms.embedding_client import EmbeddingClient

        embedding_client = EmbeddingClient(device=args.embedding_device)
        print(f"Embedding device: {embedding_client.device}")

    load_paper_graph(
        graph_store,
        records,
        batch_size=args.batch_size,
        clear=not args.no_clear,
        embedding_client=embedding_client,
        embedding_batch_size=args.embedding_batch_size,
    )

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
    if args.with_embeddings:
        embedding_counts = graph_store.query(
            """
            MATCH (n)
            WHERE n.embedding IS NOT NULL
            RETURN labels(n)[0] AS label, count(n) AS count
            ORDER BY label
            """
        )
        print("Embedding node counts:")
        for row in embedding_counts:
            print(f"{row['label']}: {row['count']}")
    graph_store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
