from new_experiments.core import EvidenceUnit, PaperExperimentRunner, RetrievalConfig
from src.llms.base_client import LLMResponse


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


def test_support_fact_metrics_do_not_count_wrong_sentence_from_relevant_title():
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
                content="The parent paragraph contains the support sentence.",
                score=0.8,
                granularity="paragraph",
            ),
        ],
        relevant_facts={("Alpha", "0"), ("Beta", "2")},
        avg_context_len=10,
        latency_ms=5,
    )

    assert metrics.recall == 0.5
    assert metrics.mrr == 0.5


class FakeGraphStore:
    def __init__(self):
        self.queries = []

    def query(self, cypher, params=None):
        self.queries.append((cypher, params or {}))
        if "MATCH (s:Section" in cypher:
            return [{"title": "Alpha", "content": "Graph parent text", "paragraph_id": "paragraph::alpha::0"}]
        return []


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


def test_fixed_graph_variant_uses_same_initial_sources_and_reranker_as_adaptive():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    runner.graph_store = FakeGraphStore()
    vector = EvidenceUnit(id="sent::alpha::0", title="Alpha", content="Alpha founded Beta.", score=0.9)
    keyword = EvidenceUnit(id="kw::beta", title="Beta", content="Beta keyword page.", score=0.6)
    seen = {}
    runner.vector_retrieve = lambda query, store_name, top_k=None: [vector]
    runner.keyword_retrieve = lambda query, top_k=None: [keyword]

    def fake_rerank(query, units, top_k=None):
        seen["ids"] = [unit.id for unit in units]
        return list(units)

    runner.rerank_units = fake_rerank
    runner.graph_expand = lambda titles, hops=1, limit_per_seed=None: []

    runner.retrieve_fixed_graph("Alpha Beta", hops=1)

    assert seen["ids"] == ["sent::alpha::0", "kw::beta"]


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
    runner.graph_expand = lambda titles, hops=1, limit_per_seed=None: []

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


def test_adaptive_route_uses_multihop_indicator_as_graph_trigger():
    runner = PaperExperimentRunner(RetrievalConfig(require_neo4j=True))
    candidates = [
        EvidenceUnit(
            id="sent::alpha::0",
            title="Alpha",
            content="Alpha and Beta are both artists with related careers.",
            score=0.95,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
        EvidenceUnit(
            id="sent::alpha::1",
            title="Alpha",
            content="Alpha worked with Beta.",
            score=0.9,
            source="graph_semantic_sentence",
            granularity="sentence",
            is_sentence_level=True,
        ),
    ]

    route, detail = runner.choose_adaptive_route("Are Alpha and Beta both artists?", candidates)

    assert detail["multi_hop_indicator"] is True
    assert route == "graph_expansion"


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
