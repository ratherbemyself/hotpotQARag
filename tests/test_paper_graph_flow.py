from new_experiments.core import EvidenceUnit, PaperExperimentRunner, RetrievalConfig
from new_experiments.route_diagnostics import choose_best_route, restore_balanced_samples, summarize_records
from src.llms.base_client import LLMResponse


class FakeComplexity:
    def __init__(self, score):
        self.score = score


class FakeComplexityScorer:
    def __init__(self, scores):
        self.scores = scores

    def compute(self, query):
        return FakeComplexity(self.scores[query])


def test_sample_hotpotqa_by_complexity_balances_low_medium_high_buckets():
    import pandas as pd
    from collections import Counter

    from new_experiments.core import complexity_level_for_score, sample_hotpotqa_by_complexity

    scores = {
        "low-1": 0.1,
        "low-2": 0.2,
        "low-3": 0.3,
        "mid-1": 0.5,
        "mid-2": 0.6,
        "mid-3": 0.7,
        "high-1": 0.8,
        "high-2": 0.9,
        "high-3": 1.0,
    }
    data = pd.DataFrame({"id": list(scores), "question": list(scores)})

    sampled = sample_hotpotqa_by_complexity(
        data,
        sample_size=6,
        scorer=FakeComplexityScorer(scores),
        random_seed=42,
        complexity_threshold=0.8,
    )

    counts = Counter(complexity_level_for_score(scores[q], complexity_threshold=0.8) for q in sampled["question"])
    assert counts == {"low": 2, "medium": 2, "high": 2}


def test_sample_hotpotqa_by_complexity_matches_paper_500_each_when_total_is_1500():
    import pandas as pd
    from collections import Counter

    from new_experiments.core import complexity_level_for_score, sample_hotpotqa_by_complexity

    scores = {}
    for prefix, score in (("low", 0.2), ("medium", 0.6), ("high", 0.9)):
        for idx in range(600):
            scores[f"{prefix}-{idx}"] = score
    data = pd.DataFrame({"id": list(scores), "question": list(scores)})

    sampled = sample_hotpotqa_by_complexity(
        data,
        sample_size=1500,
        scorer=FakeComplexityScorer(scores),
        random_seed=42,
        complexity_threshold=0.8,
    )

    counts = Counter(complexity_level_for_score(scores[q], complexity_threshold=0.8) for q in sampled["question"])
    assert counts == {"low": 500, "medium": 500, "high": 500}


def test_sample_hotpotqa_by_complexity_uses_ranked_terciles_not_fixed_threshold_capacity():
    import pandas as pd
    from collections import Counter

    from new_experiments.core import sample_hotpotqa_by_complexity

    scores = {}
    for idx in range(1600):
        if idx < 700:
            score = 0.2
        elif idx < 1200:
            score = 0.6
        else:
            score = 0.9
        scores[f"q-{idx}"] = score
    data = pd.DataFrame({"id": list(scores), "question": list(scores)})

    sampled = sample_hotpotqa_by_complexity(
        data,
        sample_size=1500,
        scorer=FakeComplexityScorer(scores),
        random_seed=42,
        complexity_threshold=0.8,
    )

    assert Counter(sampled["_complexity_level"]) == {"low": 500, "medium": 500, "high": 500}


def test_prepare_paper_graph_rows_builds_structure_and_semantic_graph():
    from src.storage.graph_store.paper_graph_builder import prepare_paper_graph_rows

    rows = prepare_paper_graph_rows(
        [
            {
                "title": "Alpha",
                "sentence_total": "Alice founded Beta. Beta is based in Paris.",
                "triplets": [
                    {"Subject": "Alice", "Predicate": "founded", "Object": "Beta"},
                    {"Subject": "Beta", "Predicate": "based in", "Object": "Paris"},
                ],
            }
        ]
    )

    assert rows["documents"] == [{"id": "doc::alpha", "title": "Alpha"}]
    assert rows["sections"][0]["id"] == "section::alpha"
    assert rows["paragraphs"][0]["section_id"] == "section::alpha"
    assert [s["text"] for s in rows["sentences"]] == [
        "Alice founded Beta.",
        "Beta is based in Paris.",
    ]
    assert {edge["type"] for edge in rows["structure_edges"]} >= {
        "HAS_SECTION",
        "HAS_PARAGRAPH",
        "HAS_SENTENCE",
        "NEXT_SENTENCE",
    }
    assert {entity["name"] for entity in rows["entities"]} == {"Alice", "Beta", "Paris"}
    assert {
        (edge["source_key"], edge["predicate"], edge["target_key"])
        for edge in rows["related_edges"]
    } == {
        ("alice", "founded", "beta"),
        ("beta", "based in", "paris"),
    }
    assert {link["section_id"] for link in rows["semantic_links"]} == {"section::alpha"}


def test_prepare_paper_graph_rows_preserves_punctuation_distinct_titles():
    from src.storage.graph_store.paper_graph_builder import prepare_paper_graph_rows

    rows = prepare_paper_graph_rows(
        [
            {"title": "Warriors Path State Park", "sentence_total": "One."},
            {"title": "Warriors' Path State Park", "sentence_total": "Two."},
        ]
    )

    section_ids = [row["id"] for row in rows["sections"]]
    assert len(set(section_ids)) == 2
    assert section_ids[0] == "section::warriors_path_state_park"
    assert section_ids[1].startswith("section::warriors_path_state_park::")


class TinyEmbeddingClient:
    def __init__(self):
        self.calls = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return [[float(len(text)), 1.0] for text in texts]


def test_attach_paper_graph_embeddings_adds_vectors_to_text_nodes_in_batches():
    from src.storage.graph_store.paper_graph_builder import (
        attach_paper_graph_embeddings,
        prepare_paper_graph_rows,
    )

    rows = prepare_paper_graph_rows(
        [
            {
                "title": "Alpha",
                "sentence_total": "Alice founded Beta. Beta is based in Paris.",
                "triplets": [{"Subject": "Alice", "Predicate": "founded", "Object": "Beta"}],
            }
        ]
    )
    embedding_client = TinyEmbeddingClient()

    embedded = attach_paper_graph_embeddings(rows, embedding_client, batch_size=2)

    assert embedded is rows
    assert all("embedding" in row for row in rows["sections"])
    assert all("embedding" in row for row in rows["paragraphs"])
    assert all("embedding" in row for row in rows["sentences"])
    assert rows["sections"][0]["embedding"] == [float(len("Alpha\nAlice founded Beta. Beta is based in Paris.")), 1.0]
    assert len(embedding_client.calls) == 2


def test_retrieval_metrics_deduplicate_titles_before_scoring():
    from new_experiments.core import compute_retrieval_metrics

    metrics = compute_retrieval_metrics(
        retrieved_titles=["Alpha", "Alpha", "Beta"],
        relevant_titles={"Alpha", "Beta"},
        avg_context_len=10,
        latency_ms=5,
    )

    assert metrics.recall == 1.0
    assert metrics.precision == 1.0
    assert metrics.ndcg <= 1.0
    assert metrics.map_score <= 1.0


def test_support_fact_metrics_do_not_count_wrong_sentence_or_wrong_paragraph_from_relevant_title():
    from new_experiments.core import compute_support_fact_metrics

    metrics = compute_support_fact_metrics(
        retrieved_units=[
            EvidenceUnit(
                id="sent::alpha::1",
                title="Alpha",
                content="A non-supporting sentence.",
                score=0.9,
                is_sentence_level=True,
                sentence_id="1",
            ),
            EvidenceUnit(
                id="parent::beta",
                title="Beta",
                content="The parent paragraph is from the relevant title but not the support sentence.",
                score=0.8,
                granularity="paragraph",
                metadata={"sentence_ids": ["1"]},
            ),
        ],
        relevant_facts={("Alpha", "0"), ("Beta", "2")},
        avg_context_len=10,
        latency_ms=5,
    )

    assert metrics.recall == 0.0
    assert metrics.mrr == 0.0


def test_support_fact_metrics_count_parent_paragraph_when_it_contains_support_sentence_id():
    from new_experiments.core import compute_support_fact_metrics

    metrics = compute_support_fact_metrics(
        retrieved_units=[
            EvidenceUnit(
                id="parent::beta",
                title="Beta",
                content="The parent paragraph contains the support sentence.",
                score=0.8,
                granularity="paragraph",
                metadata={"sentence_ids": ["2", "3"]},
            ),
        ],
        relevant_facts={("Beta", "2")},
        avg_context_len=10,
        latency_ms=5,
    )

    assert metrics.recall == 1.0
    assert metrics.mrr == 1.0


def test_paper_table_metrics_use_support_fact_labels_for_ranking_metrics():
    from new_experiments.core import ExperimentMetrics, compute_paper_table_metrics

    support_metrics = ExperimentMetrics(
        recall=0.5,
        precision=0.25,
        mrr=0.2,
        ndcg=0.3,
        map_score=0.4,
        avg_len=123.0,
        time_ms=45.0,
        expanded_nodes=6.0,
    )
    title_metrics = ExperimentMetrics(
        recall=1.0,
        precision=1.0,
        mrr=0.75,
        ndcg=0.8,
        map_score=0.7,
        avg_len=999.0,
        time_ms=999.0,
        expanded_nodes=999.0,
    )

    paper_metrics = compute_paper_table_metrics(support_metrics, title_metrics)

    assert paper_metrics.recall == 0.5
    assert paper_metrics.precision == 0.25
    assert paper_metrics.mrr == 0.2
    assert paper_metrics.ndcg == 0.3
    assert paper_metrics.map_score == 0.4
    assert paper_metrics.avg_len == 123.0
    assert paper_metrics.time_ms == 45.0
    assert paper_metrics.expanded_nodes == 6.0


def test_summary_uses_support_fact_metrics_when_paper_metrics_present():
    from new_experiments.core import ExperimentMetrics

    runner = PaperExperimentRunner(RetrievalConfig())
    strict_metrics = ExperimentMetrics(recall=0.5, precision=0.25, mrr=0.2, ndcg=0.3, map_score=0.4)
    paper_metrics = ExperimentMetrics(recall=0.5, precision=0.25, mrr=0.75, ndcg=0.8, map_score=0.7)

    summary = runner.aggregate_method_rows(
        [
            {
                "retrieval_metrics": strict_metrics,
                "paper_metrics": paper_metrics,
                "semantic_metrics": {},
            }
        ]
    )

    assert summary["Recall"] == 0.5
    assert summary["Precision"] == 0.25
    assert summary["MRR"] == 0.2
    assert summary["NDCG"] == 0.3
    assert summary["MAP"] == 0.4


class FakeGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        if "MATCH (s:Section" in cypher:
            return [{"title": "Alpha", "content": "Graph parent text", "paragraph_id": "paragraph::alpha::0"}]
        return []


class ExpansionGraphStore:
    def __init__(self, content, sentences=None):
        self.content = content
        self.sentences = sentences or []
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        return [{"title": "Beta", "content": self.content, "sentences": self.sentences}]


class MultiExpansionGraphStore:
    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        limit = int((params or {}).get("limit", len(self.rows)))
        return self.rows[:limit]


class GraphGateReranker:
    def __init__(self):
        self.calls = []

    def rerank(self, query, candidates, top_k=None):
        self.calls.append((query, candidates, top_k))
        candidates = sorted(
            candidates,
            key=lambda candidate: (
                "strong bridge" not in candidate.content,
                candidate.content,
            ),
        )
        for index, candidate in enumerate(candidates):
            candidate.score = float(len(candidates) - index)
        return candidates[:top_k] if top_k is not None else candidates


class ParentBySentenceGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        params = params or {}
        self.queries.append((cypher, params))
        if "target:Sentence" in cypher:
            assert params["title"] == "Alpha"
            assert params["sentence_id"] == "7"
            return [
                {
                    "title": "Alpha",
                    "content": "The exact parent paragraph for sentence seven.",
                    "paragraph_id": "paragraph::alpha::2",
                    "sentence_ids": ["6", "7", "8"],
                }
            ]
        return []


class StrictParentFallbackGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        if "ORDER BY p.position" in cypher and "WITH s, p" not in cypher:
            raise AssertionError("fallback parent lookup must order paragraphs before aggregating sentence ids")
        return [
            {
                "title": "Alpha",
                "content": "First parent paragraph.",
                "paragraph_id": "paragraph::alpha::0",
                "sentence_ids": ["0"],
            }
        ]


def test_parent_map_uses_neo4j_structure_graph_not_json_fallback():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=False))
    runner.graph_store = FakeGraphStore()
    runner.title_to_content = {"Alpha": "JSON parent text"}

    mapped = runner.parent_map(
        [
            EvidenceUnit(
                id="sent::alpha::0",
                title="Alpha",
                content="Graph",
                score=0.8,
                is_sentence_level=True,
                sentence_id="0",
            )
        ]
    )

    assert mapped[0].content == "Graph parent text"
    assert mapped[0].metadata["paragraph_id"] == "paragraph::alpha::0"
    assert runner.graph_store.queries


def test_parent_map_fetches_parent_paragraph_containing_trigger_sentence():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = ParentBySentenceGraphStore()

    mapped = runner.parent_map(
        [
            EvidenceUnit(
                id="sent::alpha::7",
                title="Alpha",
                content="Trigger sentence.",
                score=0.8,
                is_sentence_level=True,
                sentence_id="7",
            )
        ]
    )

    assert mapped[0].content == "The exact parent paragraph for sentence seven."
    assert mapped[0].metadata["paragraph_id"] == "paragraph::alpha::2"
    assert mapped[0].metadata["sentence_ids"] == ["6", "7", "8"]
    assert mapped[0].metadata["trigger_sentence_ids"] == ["7"]


def test_parent_lookup_fallback_orders_paragraphs_before_sentence_id_aggregation():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = StrictParentFallbackGraphStore()

    parent = runner.fetch_parent_from_graph("Alpha")

    assert parent["content"] == "First parent paragraph."
    assert parent["sentence_ids"] == ["0"]


def test_graph_expand_returns_query_focused_snippet_not_full_section_text():
    content = (
        "Opening noise unrelated to the question. "
        "Alpha Beta bridge fact gives the useful evidence. "
        "More unrelated background that should not be copied in full."
    )
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = ExpansionGraphStore(content)

    units = runner.graph_expand(["Alpha"], hops=1, limit_per_seed=5, query="Alpha Beta")

    assert len(units) == 1
    assert units[0].title == "Beta"
    assert units[0].content == "Alpha Beta bridge fact gives the useful evidence."
    assert units[0].content != content
    assert runner.graph_store.queries[0][1]["limit"] == runner.config.max_graph_neighbors


def test_graph_expand_tracks_selected_snippet_sentence_ids_for_support_metrics():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = ExpansionGraphStore(
        content="Opening noise. Alpha Beta bridge fact gives the useful evidence. Closing noise.",
        sentences=[
            {"id": 0, "text": "Opening noise."},
            {"id": 1, "text": "Alpha Beta bridge fact gives the useful evidence."},
            {"id": 2, "text": "Closing noise."},
        ],
    )

    units = runner.graph_expand(["Alpha"], hops=1, limit_per_seed=5, query="Alpha Beta")

    assert units[0].content == "Alpha Beta bridge fact gives the useful evidence."
    assert units[0].metadata["sentence_ids"] == ["1"]


def test_graph_expand_deduplicates_repeated_sentence_rows_from_multiple_semantic_paths():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = ExpansionGraphStore(
        content="Alpha Beta bridge fact gives the useful evidence. Closing noise.",
        sentences=[
            {"id": 1, "text": "Alpha Beta bridge fact gives the useful evidence."},
            {"id": 1, "text": "Alpha Beta bridge fact gives the useful evidence."},
            {"id": 2, "text": "Closing noise."},
        ],
    )

    units = runner.graph_expand(["Alpha"], hops=1, limit_per_seed=5, query="Alpha Beta")

    assert units[0].content == "Alpha Beta bridge fact gives the useful evidence."
    assert units[0].metadata["sentence_ids"] == ["1"]


def test_graph_expand_uses_candidate_pool_before_final_gate_limit():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True, max_graph_neighbors=3))
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": "Noise",
                "content": "Unrelated page about geography.",
                "sentences": [{"id": 0, "text": "Unrelated page about geography."}],
                "path_entities": [["Common"]],
            },
            {
                "title": "Beta",
                "content": "Alpha Beta bridge fact gives the useful evidence.",
                "sentences": [{"id": 1, "text": "Alpha Beta bridge fact gives the useful evidence."}],
                "path_entities": [["Alpha", "Beta"]],
            },
        ]
    )

    units = runner.graph_expand(["Alpha"], hops=1, limit_per_seed=1, query="Alpha Beta")

    assert [unit.title for unit in units] == ["Beta"]
    assert runner.graph_store.queries[0][1]["limit"] == 3


def test_graph_expand_overfetches_then_filters_noise_before_final_limit():
    runner = PaperExperimentRunner(
        RetrievalConfig(
            require_neo4j=True,
            max_graph_neighbors=4,
            graph_expansion_limit_per_seed=1,
        )
    )
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": "Noise",
                "content": "Unrelated page about geography.",
                "sentences": [{"id": 0, "text": "Unrelated page about geography."}],
                "path_entities": [["Common"]],
            },
            {
                "title": "Beta",
                "content": "Beta collaborated with Alpha on the bridge project.",
                "sentences": [{"id": 1, "text": "Beta collaborated with Alpha on the bridge project."}],
                "path_entities": [["Alpha", "Beta"]],
            },
        ]
    )
    seed = EvidenceUnit(
        id="sent::alpha::0",
        title="Alpha",
        content="Alpha collaborated with an unnamed partner.",
        score=0.9,
        is_sentence_level=True,
    )

    units = runner.graph_expand(
        ["Alpha"],
        hops=1,
        limit_per_seed=1,
        query="Who collaborated with Alpha?",
        seed_units=[seed],
    )

    assert [unit.title for unit in units] == ["Beta"]
    assert runner.graph_store.queries[0][1]["limit"] == 4
    assert units[0].metadata["graph_gate"]["decision"] == "kept"
    assert units[0].metadata["graph_gate"]["path_entity_overlap"] > 0


def test_graph_expand_reranks_filtered_candidates_with_seed_context():
    runner = PaperExperimentRunner(
        RetrievalConfig(
            require_neo4j=True,
            max_graph_neighbors=4,
            graph_expansion_limit_per_seed=2,
        )
    )
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": "Weak",
                "content": "Weak bridge mentions Alpha but lacks the decisive relation.",
                "sentences": [{"id": 0, "text": "Weak bridge mentions Alpha but lacks the decisive relation."}],
                "path_entities": [["Alpha", "Weak"]],
            },
            {
                "title": "Strong",
                "content": "Strong bridge evidence says Alpha collaborated with Beta.",
                "sentences": [{"id": 1, "text": "Strong bridge evidence says Alpha collaborated with Beta."}],
                "path_entities": [["Alpha", "Strong"]],
            },
        ]
    )
    reranker = GraphGateReranker()
    runner.reranker = reranker
    seed = EvidenceUnit(
        id="sent::alpha::0",
        title="Alpha",
        content="Alpha collaborated with someone in the archive.",
        score=0.9,
        is_sentence_level=True,
    )

    units = runner.graph_expand(
        ["Alpha"],
        hops=1,
        limit_per_seed=2,
        query="Who collaborated with Alpha?",
        seed_units=[seed],
    )

    assert [unit.title for unit in units] == ["Strong", "Weak"]
    assert "Initial evidence" in reranker.calls[0][0]
    assert "Alpha collaborated with someone" in reranker.calls[0][0]
    assert reranker.calls[0][2] == 2


def test_graph_expand_reranks_all_seed_candidates_in_one_global_pass():
    runner = PaperExperimentRunner(
        RetrievalConfig(
            require_neo4j=True,
            max_graph_neighbors=2,
            graph_expansion_limit_per_seed=1,
        )
    )
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": "Weak",
                "content": "Weak bridge mentions Alpha but lacks the decisive relation.",
                "sentences": [{"id": 0, "text": "Weak bridge mentions Alpha but lacks the decisive relation."}],
                "path_entities": [["Alpha", "Weak"]],
            },
            {
                "title": "Strong",
                "content": "Strong bridge evidence says Alpha collaborated with Beta.",
                "sentences": [{"id": 1, "text": "Strong bridge evidence says Alpha collaborated with Beta."}],
                "path_entities": [["Beta", "Strong"]],
            },
        ]
    )
    reranker = GraphGateReranker()
    runner.reranker = reranker
    seeds = [
        EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha has a weak lead.", score=0.9),
        EvidenceUnit(id="sent::beta::0", title="Beta", content="Beta has a strong bridge clue.", score=0.8),
    ]

    units = runner.graph_expand(
        ["Alpha", "Beta"],
        hops=1,
        limit_per_seed=1,
        query="Who collaborated with Alpha and Beta?",
        seed_units=seeds,
    )

    assert len(reranker.calls) == 1
    assert reranker.calls[0][2] == 2
    assert [unit.title for unit in units] == ["Strong", "Weak"]


def test_graph_expand_source_mode_uses_exact_limit_and_skips_gate_rerank():
    runner = PaperExperimentRunner(
        RetrievalConfig(
            require_neo4j=True,
            max_graph_neighbors=4,
            graph_expansion_limit_per_seed=1,
        )
    )
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": "Noise",
                "content": "Unrelated page about geography.",
                "sentences": [{"id": 0, "text": "Unrelated page about geography."}],
                "path_entities": [["Common"]],
            },
            {
                "title": "Beta",
                "content": "Alpha Beta bridge fact gives the useful evidence.",
                "sentences": [{"id": 1, "text": "Alpha Beta bridge fact gives the useful evidence."}],
                "path_entities": [["Alpha", "Beta"]],
            },
        ]
    )
    reranker = GraphGateReranker()
    runner.reranker = reranker

    units = runner.graph_expand(
        ["Alpha"],
        hops=1,
        limit_per_seed=1,
        query="Alpha Beta",
        apply_gate=False,
    )

    assert [unit.title for unit in units] == ["Noise"]
    assert runner.graph_store.queries[0][1]["limit"] == 1
    assert reranker.calls == []
    assert "graph_gate" not in units[0].metadata


def test_graph_expand_caps_rerank_pool_before_cross_encoder():
    runner = PaperExperimentRunner(
        RetrievalConfig(
            require_neo4j=True,
            max_graph_neighbors=5,
            graph_expansion_limit_per_seed=5,
            graph_rerank_candidate_cap=2,
        )
    )
    runner.graph_store = MultiExpansionGraphStore(
        [
            {
                "title": f"Candidate {idx}",
                "content": f"Candidate {idx} mentions Alpha Beta bridge evidence.",
                "sentences": [{"id": idx, "text": f"Candidate {idx} mentions Alpha Beta bridge evidence."}],
                "path_entities": [["Alpha", "Beta", f"Candidate {idx}"]],
            }
            for idx in range(5)
        ]
    )
    reranker = GraphGateReranker()
    runner.reranker = reranker

    units = runner.graph_expand(["Alpha"], hops=1, limit_per_seed=5, query="Alpha Beta")

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 2
    assert len(units) == 2


def test_parent_map_aggregates_trigger_sentence_positions_for_same_parent():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = FakeGraphStore()

    mapped = runner.parent_map(
        [
            EvidenceUnit(
                id="sent::alpha::0",
                title="Alpha",
                content="First",
                score=0.8,
                is_sentence_level=True,
                sentence_id="0",
            ),
            EvidenceUnit(
                id="sent::alpha::1",
                title="Alpha",
                content="Second",
                score=0.7,
                is_sentence_level=True,
                sentence_id="1",
            ),
        ]
    )

    assert len(mapped) == 1
    assert mapped[0].metadata["trigger_sentence_ids"] == ["0", "1"]
    assert mapped[0].metadata["trigger_unit_ids"] == ["sent::alpha::0", "sent::alpha::1"]


def test_local_parent_route_keeps_initial_evidence_and_adds_parent_context():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = FakeGraphStore()
    seed = EvidenceUnit(
        id="sent::alpha::0",
        title="Alpha",
        content="Alpha founded Beta.",
        score=0.95,
        source="graph_semantic_sentence",
        granularity="sentence",
        is_sentence_level=True,
        sentence_id="0",
    )
    runner.vector_retrieve = lambda query, store_name, top_k=None: [seed]
    runner.keyword_retrieve = lambda query, top_k=None: []
    runner.rerank_units = lambda query, units, top_k=None: list(units)
    runner.choose_adaptive_route = lambda query, candidates: (
        "local_parent",
        {
            "complexity_score": 0.2,
            "complementarity": 1.0,
            "route": "local_parent",
        },
    )

    result = runner.retrieve_adaptive("Who founded Alpha?")

    assert [unit.id for unit in result.units] == ["sent::alpha::0", "parent::Alpha"]
    assert result.units[1].content == "Graph parent text"
    assert result.units[1].metadata["trigger_sentence_ids"] == ["0"]


def test_fixed_graph_variant_uses_source_fixed_scale_candidate_flow():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = FakeGraphStore()
    vectors = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Seed {idx}", content=f"Seed {idx} evidence.", score=1.0 - idx / 10)
        for idx in range(5)
    ]
    seen = {}

    def fake_vector(query, store_name, top_k=None):
        seen["vector"] = {"query": query, "store_name": store_name, "top_k": top_k}
        return vectors

    runner.vector_retrieve = fake_vector
    runner.keyword_retrieve = lambda query, top_k=None: (_ for _ in ()).throw(
        AssertionError("source fixed-scale graph variants use only sentence vector candidates")
    )

    def fake_rerank(query, units, top_k=None):
        seen["ids"] = [unit.id for unit in units]
        seen["top_k"] = top_k
        return list(units)

    runner.rerank_units = fake_rerank

    def fake_graph_expand(titles, hops=1, limit_per_seed=None, query="", use_snippet=True, apply_gate=True, seed_units=None):
        seen["titles"] = list(titles)
        seen["hops"] = hops
        seen["limit"] = limit_per_seed
        seen["query"] = query
        seen["use_snippet"] = use_snippet
        seen["apply_gate"] = apply_gate
        return []

    runner.graph_expand = fake_graph_expand

    runner.retrieve_fixed_graph("Alpha Beta", hops=1)

    assert seen["vector"] == {"query": "Alpha Beta", "store_name": "sentence", "top_k": runner.config.k1}
    assert seen["ids"] == [unit.id for unit in vectors]
    assert seen["top_k"] == runner.config.k3
    assert seen["titles"] == ["Seed 0", "Seed 1", "Seed 2"]
    assert seen["hops"] == 1
    assert seen["limit"] == runner.config.graph_expansion_limit_per_seed
    assert seen["query"] == "Alpha Beta"
    assert seen["use_snippet"] is False
    assert seen["apply_gate"] is False


def test_fixed_graph_variant_does_not_mix_in_parent_scale():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    seeds = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Seed {idx}", content=f"Seed {idx} evidence.", score=1.0 - idx / 10)
        for idx in range(runner.config.k3)
    ]
    expanded = EvidenceUnit(id="graph::beta", title="Beta", content="Beta expanded.", score=0.5)
    runner.vector_retrieve = lambda query, store_name, top_k=None: seeds
    runner.keyword_retrieve = lambda query, top_k=None: []
    runner.rerank_units = lambda query, units, top_k=None: list(units)
    runner.graph_expand = lambda titles, hops=1, limit_per_seed=None, query="", use_snippet=True, apply_gate=True, seed_units=None: [expanded]
    runner.parent_context_pool = lambda units: (_ for _ in ()).throw(
        AssertionError("fixed graph variants should isolate graph expansion from parent scale")
    )
    runner.select_with_budget = lambda query, units, max_units=None: (_ for _ in ()).throw(
        AssertionError("source fixed-scale graph variants slice reranked+expanded directly")
    )

    result = runner.retrieve_fixed_graph("Alpha Beta", hops=1)

    assert [unit.id for unit in result.units] == [unit.id for unit in seeds] + ["graph::beta"]


def test_route_diagnostics_best_route_prefers_recall_then_lower_token_cost():
    rows = [
        {"method": "fine_grained", "recall": 0.5, "avg_len": 200, "expanded_nodes": 0},
        {"method": "local_parent", "recall": 1.0, "avg_len": 900, "expanded_nodes": 0},
        {"method": "graph_expansion", "recall": 1.0, "avg_len": 1600, "expanded_nodes": 8},
        {"method": "proposed", "recall": 1.0, "avg_len": 1600, "expanded_nodes": 8},
    ]

    assert choose_best_route(rows) == "local_parent"


def test_route_diagnostics_summary_groups_by_original_route_and_method():
    rows = [
        {
            "original_route": "graph_expansion",
            "method": "fine_grained",
            "recall": 0.5,
            "precision": 0.2,
            "avg_len": 100,
            "expanded_nodes": 0,
            "time_ms": 10,
        },
        {
            "original_route": "graph_expansion",
            "method": "fine_grained",
            "recall": 1.0,
            "precision": 0.4,
            "avg_len": 300,
            "expanded_nodes": 0,
            "time_ms": 30,
        },
    ]

    assert summarize_records(rows) == [
        {
            "original_route": "graph_expansion",
            "method": "fine_grained",
            "n": 2,
            "recall": 0.75,
            "precision": 0.3,
            "avg_len": 200.0,
            "expanded_nodes": 0.0,
            "time_ms": 20.0,
        }
    ]


def test_route_diagnostics_restores_balanced_csv_rows_from_source_data():
    import pandas as pd

    source = pd.DataFrame(
        [
            {
                "id": "alpha",
                "question": "Who founded Alpha?",
                "supporting_facts": {"title": ["Alpha"], "sent_id": [0]},
                "context": {"title": ["Alpha"], "sentences": [["Alpha was founded by Beta."]]},
            },
            {
                "id": "beta",
                "question": "Who founded Beta?",
                "supporting_facts": {"title": ["Beta"], "sent_id": [0]},
                "context": {"title": ["Beta"], "sentences": [["Beta was founded by Gamma."]]},
            },
        ]
    )
    csv_rows = pd.DataFrame(
        [
            {
                "id": "beta",
                "question": "stale csv question",
                "supporting_facts": "{'title': ['wrong']}",
                "_diagnostic_route": "graph_expansion",
                "_route_detail": '{"route":"graph_expansion"}',
                "_scan_order": 7,
            }
        ]
    )

    restored = restore_balanced_samples(source, csv_rows)

    assert restored.iloc[0]["question"] == "Who founded Beta?"
    assert restored.iloc[0]["supporting_facts"] == {"title": ["Beta"], "sent_id": [0]}
    assert restored.iloc[0]["_diagnostic_route"] == "graph_expansion"
    assert restored.iloc[0]["_scan_order"] == 7


def test_adaptive_graph_route_expands_the_full_reranked_b0_with_controlled_neighbor_limit():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    seeds = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Seed {idx}", content=f"Seed {idx} evidence.", score=1.0 - idx / 10)
        for idx in range(7)
    ]
    seen = {}
    runner.vector_retrieve = lambda query, store_name, top_k=None: seeds
    runner.keyword_retrieve = lambda query, top_k=None: []
    runner.rerank_units = lambda query, units, top_k=None: list(units)
    runner.choose_adaptive_route = lambda query, candidates: (
        "graph_expansion",
        {
            "complexity_score": 0.9,
            "complementarity": 0.2,
            "route": "graph_expansion",
        },
    )

    def fake_graph_expand(titles, hops=1, limit_per_seed=None, query="", use_snippet=True, seed_units=None):
        seen["titles"] = list(titles)
        seen["limit"] = limit_per_seed
        seen["query"] = query
        seen["use_snippet"] = use_snippet
        seen["seed_units"] = list(seed_units or [])
        return []

    runner.graph_expand = fake_graph_expand

    runner.retrieve_adaptive("Alpha Beta bridge question")

    assert seen["titles"] == ["Seed 0", "Seed 1", "Seed 2", "Seed 3", "Seed 4", "Seed 5", "Seed 6"]
    assert seen["limit"] == 5
    assert seen["query"] == "Alpha Beta bridge question"
    assert seen["use_snippet"] is True
    assert seen["seed_units"] == seeds


def test_graphrag_baseline_uses_source_seed_window_graph_expansion():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    seeds = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Seed {idx}", content=f"Seed {idx} evidence.", score=1.0 - idx / 10)
        for idx in range(7)
    ]
    seen = {}
    runner.vector_retrieve = lambda query, store_name, top_k=None: seeds
    runner.select_with_budget = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("source GraphRAG slices initial+expanded directly")
    )

    def fake_graph_expand(titles, hops=1, limit_per_seed=None, query="", seed_limit=None, use_snippet=True, apply_gate=True):
        seen["titles"] = list(titles)
        seen["hops"] = hops
        seen["limit"] = limit_per_seed
        seen["seed_limit"] = seed_limit
        seen["use_snippet"] = use_snippet
        seen["apply_gate"] = apply_gate
        return []

    runner.graph_expand = fake_graph_expand

    runner.retrieve_graphrag("Alpha Beta")

    assert seen["titles"] == ["Seed 0", "Seed 1", "Seed 2"]
    assert seen["hops"] == 1
    assert seen["limit"] == runner.config.graph_expansion_limit_per_seed
    assert seen["seed_limit"] == 3
    assert seen["use_snippet"] is False
    assert seen["apply_gate"] is False


def test_graph_route_summary_gate_counts_parent_and_expansion_not_initial_candidates():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True, context_budget=3600))
    seed = EvidenceUnit(
        id="sent::alpha::0",
        title="Alpha",
        content="Alpha " * 9000,
        score=0.95,
        source="graph_semantic_sentence",
        granularity="sentence",
        is_sentence_level=True,
        sentence_id="0",
    )
    called = {}
    runner.vector_retrieve = lambda query, store_name, top_k=None: [seed]
    runner.keyword_retrieve = lambda query, top_k=None: []
    runner.rerank_units = lambda query, units, top_k=None: list(units)
    runner.choose_adaptive_route = lambda query, candidates: (
        "graph_expansion",
        {
            "complexity_score": 0.9,
            "complementarity": 0.2,
            "route": "graph_expansion",
        },
    )
    runner.graph_expand = lambda titles, hops=1, limit_per_seed=None, query="", seed_units=None: []

    def fake_summary(titles, query=""):
        called["titles"] = list(titles)
        return []

    runner.add_summary_evidence = fake_summary

    runner.retrieve_adaptive("Alpha bridge question", enable_parent=False)

    assert called["titles"] == ["Alpha"]


class EmptyGraphStore:
    def query(self, cypher, params=None):
        return []


def test_parent_map_does_not_use_json_fallback_when_neo4j_required():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = EmptyGraphStore()
    runner.title_to_content = {"Alpha": "JSON parent text that should not be used"}

    mapped = runner.parent_map(
        [
            EvidenceUnit(
                id="sent::alpha::0",
                title="Alpha",
                content="Sentence text",
                score=0.8,
                is_sentence_level=True,
                sentence_id="0",
            )
        ]
    )

    assert mapped[0].content == "Sentence text"
    assert mapped[0].id == "sent::alpha::0"


def test_summary_evidence_does_not_use_json_fallback_when_neo4j_required():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = EmptyGraphStore()
    runner.title_to_content = {"Alpha": "JSON summary text that should not be used"}

    assert runner.add_summary_evidence(["Alpha"], query="Alpha") == []


class SemanticGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        if "MATCH (sec:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)-[:HAS_SENTENCE]->(sent:Sentence)" in cypher:
            return [
                {
                    "id": "sentence::alpha::0",
                    "title": "Alpha",
                    "content": "Alpha founded Beta.",
                    "sent_id": 0,
                    "paragraph_id": "paragraph::alpha::0",
                },
                {
                    "id": "sentence::gamma::0",
                    "title": "Gamma",
                    "content": "Gamma is unrelated.",
                    "sent_id": 0,
                    "paragraph_id": "paragraph::gamma::0",
                },
            ]
        if "MATCH (sec:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)" in cypher:
            return [
                {
                    "id": "paragraph::alpha::0",
                    "title": "Alpha",
                    "content": "Alpha founded Beta. Beta is based in Paris.",
                    "position": 0,
                }
            ]
        return []


class LexicalEmbeddingClient:
    dimension = 3

    def embed(self, texts, normalize=True):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append(
                [
                    1.0 if "alpha" in lower else 0.0,
                    1.0 if "beta" in lower or "paris" in lower else 0.0,
                    1.0 if "gamma" in lower else 0.0,
                ]
            )
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)

    def batch_similarity(self, query, texts):
        query_terms = set(query.lower().split())
        scores = []
        for text in texts:
            terms = set(text.lower().replace(".", "").split())
            scores.append(len(query_terms & terms) / max(len(query_terms), 1))
        return scores


class CountingEmbeddingClient:
    dimension = 2

    def __init__(self):
        self.batch_sizes = []

    def embed(self, texts, normalize=True):
        import numpy as np

        if isinstance(texts, str):
            texts = [texts]
        self.batch_sizes.append(len(texts))
        vectors = []
        for text in texts:
            lower = text.lower()
            vectors.append([1.0 if "alpha" in lower or "beta" in lower else 0.0, 1.0 if "gamma" in lower else 0.0])
        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / (norms + 1e-8)


class StoredEmbeddingGraphStore:
    def query(self, cypher, params=None):
        if "MATCH (sec:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)-[:HAS_SENTENCE]->(sent:Sentence)" in cypher:
            return [
                {
                    "id": "sentence::alpha::0",
                    "title": "Alpha",
                    "content": "Alpha founded Beta.",
                    "sent_id": 0,
                    "paragraph_id": "paragraph::alpha::0",
                    "embedding": [1.0, 0.0],
                },
                {
                    "id": "sentence::gamma::0",
                    "title": "Gamma",
                    "content": "Gamma is unrelated.",
                    "sent_id": 0,
                    "paragraph_id": "paragraph::gamma::0",
                    "embedding": [0.0, 1.0],
                },
            ]
        return []


class QueryOnlyEmbeddingClient:
    dimension = 2

    def __init__(self):
        self.calls = []

    def embed(self, texts, normalize=True):
        import numpy as np

        self.calls.append(texts)
        if not isinstance(texts, str):
            raise AssertionError("candidate evidence embeddings should come from Neo4j")
        return np.array([[1.0, 0.0]], dtype=np.float32)


class VectorIndexGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        if "SHOW INDEXES" in cypher:
            return [{"name": "sentence_embedding", "type": "VECTOR", "state": "ONLINE"}]
        if "db.index.vector.queryNodes" in cypher:
            raise AssertionError("Neo4j 2026 vector retrieval should use SEARCH, not deprecated queryNodes")
        if "SEARCH node IN" in cypher and "VECTOR INDEX sentence_embedding" in cypher:
            assert (params or {}).get("limit") == 1
            assert list((params or {}).get("embedding")) == [1.0, 0.0]
            return [
                {
                    "id": "sentence::alpha::0",
                    "title": "Alpha",
                    "content": "Alpha founded Beta.",
                    "sent_id": 0,
                    "paragraph_id": "paragraph::alpha::0",
                    "score": 0.99,
                }
            ]
        if "sent.embedding AS embedding" in cypher:
            raise AssertionError("full graph embedding scan should not run when vector index answers")
        return []


def test_graph_semantic_retrieve_uses_neo4j_vector_index_without_full_embedding_scan():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = VectorIndexGraphStore()
    runner.embedding_client = QueryOnlyEmbeddingClient()

    units = runner.graph_semantic_retrieve("Alpha Beta", "sentence", top_k=1)

    assert len(units) == 1
    assert units[0].title == "Alpha"
    assert units[0].score == 0.99
    assert units[0].source == "graph_semantic_sentence"
    assert units[0].metadata["section_source"] == "neo4j"
    assert runner.embedding_client.calls == ["Alpha Beta"]
    assert runner._graph_unit_cache == {}
    assert runner._graph_embedding_cache == {}


def test_graph_semantic_retrieve_uses_stored_neo4j_embeddings_before_encoding_text():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = StoredEmbeddingGraphStore()
    runner.embedding_client = QueryOnlyEmbeddingClient()

    units = runner.graph_semantic_retrieve("Alpha Beta", "sentence", top_k=1)

    assert units[0].title == "Alpha"
    assert runner.embedding_client.calls == ["Alpha Beta"]


def test_vector_retrieve_uses_neo4j_sentence_nodes_when_neo4j_required():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = SemanticGraphStore()
    runner.embedding_client = LexicalEmbeddingClient()
    runner.sentence_store = None

    units = runner.vector_retrieve("Alpha Beta", "sentence", top_k=1)

    assert len(units) == 1
    assert units[0].title == "Alpha"
    assert units[0].content == "Alpha founded Beta."
    assert units[0].source == "graph_semantic_sentence"
    assert units[0].metadata["section_source"] == "neo4j"


def test_vector_retrieve_uses_neo4j_paragraph_nodes_when_neo4j_required():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = SemanticGraphStore()
    runner.embedding_client = LexicalEmbeddingClient()
    runner.paragraph_store = None

    units = runner.vector_retrieve("Paris Beta", "paragraph", top_k=1)

    assert len(units) == 1
    assert units[0].title == "Alpha"
    assert units[0].content == "Alpha founded Beta. Beta is based in Paris."
    assert units[0].source == "graph_semantic_paragraph"
    assert units[0].metadata["section_source"] == "neo4j"


def test_semantic_rag_baseline_matches_source_file_direct_topk():
    runner = PaperExperimentRunner(RetrievalConfig())
    seen = {}
    units = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Title {idx}", content=f"Evidence {idx}", score=1.0 - idx / 10)
        for idx in range(10)
    ]

    def fake_vector_retrieve(query, store_name="sentence", top_k=None):
        seen["store_name"] = store_name
        seen["top_k"] = top_k
        return units[:top_k]

    runner.vector_retrieve = fake_vector_retrieve

    result = runner.retrieve_semantic_rag("Alpha Beta")

    assert seen == {"store_name": "sentence", "top_k": runner.config.k3}
    assert [unit.id for unit in result.units] == [f"sent::{idx}" for idx in range(runner.config.k3)]
    assert result.stats["route"] == "semantic"


def test_rerank_baseline_matches_source_file_without_title_deduplication():
    runner = PaperExperimentRunner(RetrievalConfig())
    candidates = [
        EvidenceUnit(id="sent::alpha::0", title="Alpha", content="First Alpha evidence.", score=0.9),
        EvidenceUnit(id="sent::alpha::1", title="Alpha", content="Second Alpha evidence.", score=0.8),
    ]
    seen = {}
    runner.vector_retrieve = lambda query, store_name="sentence", top_k=None: candidates

    def fake_rerank(query, units, top_k=None):
        seen["ids"] = [unit.id for unit in units]
        seen["top_k"] = top_k
        return list(units)

    runner.rerank_units = fake_rerank

    result = runner.retrieve_rerank_rag("Alpha")

    assert seen == {"ids": ["sent::alpha::0", "sent::alpha::1"], "top_k": runner.config.k3}
    assert [unit.id for unit in result.units] == ["sent::alpha::0", "sent::alpha::1"]
    assert result.stats["route"] == "rerank"


def test_graphrag_baseline_matches_source_file_seed_and_topk_flow():
    runner = PaperExperimentRunner(RetrievalConfig())
    seeds = [
        EvidenceUnit(id=f"sent::{idx}", title=f"Seed {idx}", content=f"Seed {idx} evidence.", score=1.0 - idx / 10)
        for idx in range(10)
    ]
    expanded = [
        EvidenceUnit(id=f"graph::{idx}", title=f"Graph {idx}", content=f"Graph {idx} evidence.", score=0.5)
        for idx in range(5)
    ]
    seen = {}
    runner.vector_retrieve = lambda query, store_name="sentence", top_k=None: seeds
    runner.select_with_budget = lambda query, units, max_units=None: (_ for _ in ()).throw(
        AssertionError("source GraphRAG takes all_units[:k3], not the proposed budget selector")
    )

    def fake_graph_expand(titles, hops=1, limit_per_seed=None, query="", seed_limit=None, use_snippet=True, apply_gate=True):
        seen["titles"] = list(titles)
        seen["hops"] = hops
        seen["limit"] = limit_per_seed
        seen["seed_limit"] = seed_limit
        seen["query"] = query
        seen["use_snippet"] = use_snippet
        seen["apply_gate"] = apply_gate
        return expanded

    runner.graph_expand = fake_graph_expand

    result = runner.retrieve_graphrag("Alpha Beta")

    assert seen["titles"] == ["Seed 0", "Seed 1", "Seed 2"]
    assert seen["hops"] == 1
    assert seen["limit"] == runner.config.graph_expansion_limit_per_seed
    assert seen["seed_limit"] == 3
    assert seen["query"] == "Alpha Beta"
    assert seen["use_snippet"] is False
    assert seen["apply_gate"] is False
    assert [unit.id for unit in result.units] == [f"sent::{idx}" for idx in range(runner.config.k3)]
    assert result.stats["expanded_nodes"] == len(expanded)


def test_kg_rag_baseline_matches_source_file_vector_keyword_rerank_only():
    runner = PaperExperimentRunner(RetrievalConfig())
    vector = EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha sentence.", score=0.9)
    keyword = EvidenceUnit(id="kw::beta", title="Beta", content="Beta keyword paragraph.", score=0.6)
    calls = []
    seen = {}
    runner.vector_retrieve = lambda query, store_name="sentence", top_k=None: calls.append((store_name, top_k)) or [vector]
    runner.keyword_retrieve = lambda query, top_k=None: [keyword]
    runner.graph_expand = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("source KG-RAG does not use graph expansion")
    )
    runner.select_with_budget = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("source KG-RAG returns reranked top-k directly")
    )

    def fake_rerank(query, units, top_k=None):
        seen["ids"] = [unit.id for unit in units]
        seen["top_k"] = top_k
        return list(units)[:top_k]

    runner.rerank_units = fake_rerank

    result = runner.retrieve_kg_rag("Alpha Beta")

    assert calls == [("sentence", runner.config.k1)]
    assert seen == {"ids": ["sent::alpha::0", "kw::beta"], "top_k": runner.config.k3}
    assert [unit.id for unit in result.units] == ["sent::alpha::0", "kw::beta"]
    assert result.stats["expanded_nodes"] == 0
    assert result.stats["route"] == "kg_rag"


def test_macrag_baseline_matches_source_file_sentence_paragraph_rerank():
    runner = PaperExperimentRunner(RetrievalConfig())
    calls = []
    seen = {}

    def fake_vector_retrieve(query, store_name="sentence", top_k=None):
        calls.append((store_name, top_k))
        limit = top_k or 10
        return [
            EvidenceUnit(
                id=f"{store_name}-{idx}",
                title=f"{store_name.title()} Title {idx}",
                content=f"{store_name} evidence {idx}",
                score=0.9 - idx / 100,
                granularity=store_name,
                is_sentence_level=store_name == "sentence",
            )
            for idx in range(limit)
        ]

    def fake_rerank(query, units, top_k=None):
        seen["granularities"] = [unit.granularity for unit in units]
        seen["top_k"] = top_k
        return list(units)[:top_k]

    runner.vector_retrieve = fake_vector_retrieve
    runner.rerank_units = fake_rerank
    runner.complexity_scorer = type(
        "NoComplexityForMacRAG",
        (),
        {"compute": lambda self, query: (_ for _ in ()).throw(AssertionError("source MacRAG does not route by complexity"))},
    )()
    runner.parent_context_pool = lambda units: (_ for _ in ()).throw(
        AssertionError("MacRAG baseline should not use proposed parent context")
    )
    runner.select_with_budget = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("source MacRAG returns reranked top-k directly")
    )

    result = runner.retrieve_macrag("Which person founded Alpha?")

    assert calls == [("sentence", runner.config.k1), ("paragraph", 5)]
    assert seen == {"granularities": ["sentence"] * runner.config.k1 + ["paragraph"] * 5, "top_k": runner.config.k3}
    assert [unit.id for unit in result.units] == [f"sentence-{idx}" for idx in range(runner.config.k3)]
    assert result.stats["route"] == "macrag"


def test_fine_only_baseline_uses_shared_initial_candidates_without_parent():
    runner = PaperExperimentRunner(RetrievalConfig())
    seed = EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha evidence.", score=0.9)
    keyword = EvidenceUnit(id="kw::beta", title="Beta", content="Beta title evidence.", score=0.6)
    calls = []
    runner.vector_retrieve = lambda query, store_name="sentence", top_k=None: calls.append((store_name, top_k)) or [seed]
    runner.keyword_retrieve = lambda query, top_k=None: [keyword]
    runner.parent_map = lambda units: (_ for _ in ()).throw(
        AssertionError("fine-only baseline should not use parent mapping")
    )
    runner.rerank_units = lambda query, units, top_k=None: list(units)

    result = runner.retrieve_fine_only("Alpha Beta")

    assert calls == [("sentence", runner.config.k1)]
    assert [unit.id for unit in result.units] == ["sent::alpha::0", "kw::beta"]
    assert result.stats["route"] == "fine_only"


def test_uniform_parent_baseline_maps_shared_initial_candidates():
    runner = PaperExperimentRunner(RetrievalConfig())
    seed = EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha evidence.", score=0.9)
    keyword = EvidenceUnit(id="kw::beta", title="Beta", content="Beta title evidence.", score=0.6)
    parent = EvidenceUnit(id="parent::Alpha", title="Alpha", content="Alpha parent.", score=0.9, granularity="paragraph")
    calls = []
    runner.vector_retrieve = lambda query, store_name="sentence", top_k=None: calls.append((store_name, top_k)) or [seed]
    runner.keyword_retrieve = lambda query, top_k=None: [keyword]
    runner.rerank_units = lambda query, units, top_k=None: list(units)

    def fake_parent_map(units):
        assert [unit.id for unit in units] == ["sent::alpha::0", "kw::beta"]
        return [parent]

    runner.parent_map = fake_parent_map

    result = runner.retrieve_uniform_parent("Alpha Beta")

    assert calls == [("sentence", runner.config.k1)]
    assert [unit.id for unit in result.units] == ["parent::Alpha"]
    assert result.stats["route"] == "parent_all"


def test_graph_semantic_retrieve_embeds_neo4j_units_once_per_granularity():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = SemanticGraphStore()
    runner.embedding_client = CountingEmbeddingClient()

    first = runner.graph_semantic_retrieve("Alpha Beta", "sentence", top_k=1)
    second = runner.graph_semantic_retrieve("Gamma", "sentence", top_k=1)

    assert first[0].title == "Alpha"
    assert second[0].title == "Gamma"
    assert runner.embedding_client.batch_sizes == [2, 1, 1]


def test_select_assembles_core_then_bridge_then_summary():
    runner = PaperExperimentRunner(RetrievalConfig(context_budget=1000, require_neo4j=True))
    units = [
        EvidenceUnit(
            id="summary::alpha",
            title="Alpha",
            content="Alpha Beta background summary.",
            score=1.0,
            source="summary",
            granularity="summary",
        ),
        EvidenceUnit(
            id="graph::beta",
            title="Beta",
            content="Beta bridges Alpha to Gamma.",
            score=0.9,
            source="graph_1hop",
            granularity="paragraph",
        ),
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content="Alpha directly answers Beta.",
            score=0.8,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
    ]

    selected = runner.select_with_budget("Alpha Beta", units, max_units=3)

    assert [unit.id for unit in selected] == ["sent::alpha::0", "graph::beta", "summary::alpha"]


def test_select_recomputes_completion_value_against_already_selected_terms():
    runner = PaperExperimentRunner(RetrievalConfig(context_budget=1000, max_context_units=2))
    units = [
        EvidenceUnit(id="u1", title="Alpha", content="Alpha Beta direct evidence.", score=0.90),
        EvidenceUnit(id="u2", title="Alpha Copy", content="Alpha Beta repeated wording.", score=0.89),
        EvidenceUnit(id="u3", title="Gamma", content="Gamma complementary evidence.", score=0.60),
    ]

    selected = runner.select_with_budget("Alpha Beta Gamma", units, max_units=2)

    assert [unit.id for unit in selected] == ["u1", "u3"]


class FlakyAnswerLLM:
    max_retries = 3
    retry_delay = 0

    def __init__(self):
        self.calls = []

    def generate(self, messages, temperature, max_tokens):
        self.calls.append((messages, temperature, max_tokens))
        if len(self.calls) == 1:
            return LLMResponse(content="", success=False, error="timeout")
        if len(self.calls) == 2:
            return LLMResponse(content="   ", success=True)
        return LLMResponse(content="Alpha founded Beta.", success=True)


def test_generate_answer_retries_failed_and_blank_deepseek_responses():
    runner = PaperExperimentRunner(RetrievalConfig(run_generation=True))
    llm = FlakyAnswerLLM()
    runner.llm_client = llm

    answer = runner.generate_answer(
        "Who founded Alpha?",
        [EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha founded Beta.", score=1.0)],
    )

    assert answer == "Alpha founded Beta."
    assert len(llm.calls) == 3
    assert [call[1:] for call in llm.calls] == [(0.0, 256), (0.0, 512), (0.0, 1024)]


class FlakyJudgeLLM:
    max_retries = 3
    retry_delay = 0

    def __init__(self):
        self.calls = []

    def generate(self, messages, temperature, max_tokens, **kwargs):
        self.calls.append((messages, temperature, max_tokens, kwargs))
        if len(self.calls) == 1:
            return LLMResponse(content="not json", success=True)
        return LLMResponse(
            content='{"correctness": 1, "faithfulness": 0.9, "answer_relevance": 0.8, "context_relevance": 0.7}',
            success=True,
        )


def test_judge_answer_retries_until_parseable_metric_json():
    runner = PaperExperimentRunner(RetrievalConfig(run_judge=True))
    llm = FlakyJudgeLLM()
    runner.llm_client = llm

    metrics = runner.judge_answer(
        question="Who founded Alpha?",
        ground_truth="Beta",
        answer="Beta",
        units=[EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha was founded by Beta.", score=1.0)],
    )

    assert metrics == {
        "correctness": 1.0,
        "faithfulness": 0.9,
        "answer_relevance": 0.8,
        "context_relevance": 0.7,
    }
    assert len(llm.calls) == 2
    assert [(call[1], call[2]) for call in llm.calls] == [(0.0, 512), (0.0, 1024)]
    assert all(call[3]["response_format"] == {"type": "json_object"} for call in llm.calls)


class RouterLLM:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def generate(self, messages, temperature, max_tokens, **kwargs):
        self.calls.append((messages, temperature, max_tokens, kwargs))
        return LLMResponse(content=self.content, success=True)


def test_llm_router_parses_deepseek_json_route_and_records_fallback_detail():
    runner = PaperExperimentRunner(RetrievalConfig(use_llm_router=True))
    runner.llm_client = RouterLLM('{"route": "local_parent", "reason": "needs parent context"}')
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content="Alpha founded Beta.",
            score=0.9,
            is_sentence_level=True,
        )
    ]

    route, detail = runner.choose_llm_route("Who founded Alpha?", candidates)

    assert route == "local_parent"
    assert detail["router"] == "deepseek"
    assert detail["llm_reason"] == "needs parent context"
    assert runner.llm_client.calls[0][1:] == (0.0, 128, {"response_format": {"type": "json_object"}})


def test_adaptive_route_uses_fragmented_multihop_candidates_as_graph_trigger():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content="Alpha was an artist whose collaborator was named Gamma.",
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
        EvidenceUnit(
            id="sent::beta::0",
            title="Beta",
            content="Beta was an artist whose collaborator was named Delta.",
            score=0.9,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
        EvidenceUnit(
            id="sent::gamma::0",
            title="Gamma",
            content="Gamma and Delta were linked through a shared exhibition.",
            score=0.85,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
    ]

    route, detail = runner.choose_adaptive_route("Are Alpha and Beta both artists?", candidates)

    assert detail["multi_hop_indicator"] is True
    assert route == "graph_expansion"


def test_adaptive_route_uses_parent_for_complete_concentrated_multihop_candidates():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content=(
                "Alpha and Beta are both artists with related careers, shared exhibitions, "
                "documented biographies, public performances, gallery records, and clear "
                "career descriptions that directly answer the comparison question."
            ),
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
        EvidenceUnit(
            id="sent::alpha::1",
            title="Alpha",
            content=(
                "The same Alpha record states that Beta is also an artist, so the answer "
                "does not need an additional semantic graph bridge or parent expansion."
            ),
            score=0.9,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
    ]

    route, detail = runner.choose_adaptive_route("Are Alpha and Beta both artists?", candidates)

    assert detail["multi_hop_indicator"] is True
    assert detail["fragmentation"] <= 0.35
    assert route == "local_parent"


def test_adaptive_route_uses_parent_for_concentrated_but_complex_candidates():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content=(
                "Alpha and Beta are both artists with related careers, shared exhibitions, "
                "documented biographies, public performances, gallery records, chronology, "
                "nationality, genre, awards, collaborations, and multiple stated constraints."
            ),
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
        EvidenceUnit(
            id="sent::alpha::1",
            title="Alpha",
            content=(
                "The same Alpha record also describes Beta as an artist, but the question "
                "asks for several constraints that benefit from the parent context."
            ),
            score=0.9,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
    ]

    route, detail = runner.choose_adaptive_route(
        "Are Alpha and Beta both artists with the same nationality, genre, awards, collaborators, and exhibition chronology?",
        candidates,
    )

    assert detail["fragmentation"] <= 0.35
    assert detail["complexity_score"] >= 0.45
    assert route == "local_parent"


def test_adaptive_route_keeps_fine_grained_when_low_complexity_and_complete():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content=(
                "Alpha was founded by Beta in Paris, and the founding information is "
                "stated directly with enough local context for the answer."
            ),
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        )
    ]

    route, detail = runner.choose_adaptive_route("Who founded Alpha?", candidates)

    assert detail["multi_hop_indicator"] is False
    assert route == "fine_grained"


def test_adaptive_route_uses_local_parent_for_short_direct_evidence():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content="Alpha founded Beta.",
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        )
    ]

    route, detail = runner.choose_adaptive_route("Who founded Alpha?", candidates)

    assert detail["short_sentence_ratio"] == 1.0
    assert route == "local_parent"


def test_dynamic_hops_uses_tc_for_candidate_coverage_threshold():
    runner = PaperExperimentRunner(RetrievalConfig(hmax=2, complexity_threshold=0.80))

    hops = runner._dynamic_hops({"complexity_score": 0.9, "complementarity": 0.7})

    assert hops == 2


def test_select_paper_graph_records_keeps_sample_context_titles():
    from src.storage.graph_store.paper_graph_builder import select_paper_graph_records

    records = [
        {"title": "Alpha", "sentence_total": "Alpha text", "triplets": []},
        {"title": "Beta", "sentence_total": "Beta text", "triplets": []},
        {"title": "Gamma", "sentence_total": "Gamma text", "triplets": []},
    ]
    samples = [
        {
            "supporting_facts": {"title": ["Alpha"]},
            "context": {
                "title": ["Beta", "Missing"],
                "sentences": [["Beta sentence."], ["Missing sentence."]],
            },
        }
    ]

    selected = select_paper_graph_records(records, samples=samples)

    assert [row["title"] for row in selected] == ["Alpha", "Beta", "Missing"]
    assert selected[-1]["sentence_total"] == "Missing sentence."


class FakeClearGraphStore:
    def __init__(self):
        self.calls = []
        self.remaining_rels = 3
        self.remaining_nodes = 2

    def query(self, cypher, params=None):
        self.calls.append((cypher, params or {}))
        if "MATCH ()-[r]->()" in cypher and "RETURN count" in cypher:
            current = self.remaining_rels
            self.remaining_rels = 0
            return [{"count": current}]
        if "MATCH (n)" in cypher and "RETURN count" in cypher:
            current = self.remaining_nodes
            self.remaining_nodes = 0
            return [{"count": current}]
        return []

    def clear(self):
        raise AssertionError("single-transaction clear must not be used")


def test_clear_paper_graph_uses_batched_deletes_instead_of_single_clear():
    from src.storage.graph_store.paper_graph_builder import clear_paper_graph

    graph_store = FakeClearGraphStore()
    clear_paper_graph(graph_store, batch_size=100)

    all_cypher = "\n".join(cypher for cypher, _ in graph_store.calls)
    assert "LIMIT $batch_size" in all_cypher
    assert "DELETE r" in all_cypher
    assert "DELETE n" in all_cypher
