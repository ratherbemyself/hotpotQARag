# -*- coding: utf-8 -*-
"""

本文件为所有实验的共享代码。
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.logger import get_logger


logger = get_logger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOTPOT_DIR = PROJECT_ROOT / "data" / "hotpotqa"


@dataclass
class RetrievalConfig:
    """论文实验的共享配置。"""

    test_data_path: Path = DEFAULT_HOTPOT_DIR / "validation-00000-of-00001.parquet"
    documents_path: Path = DEFAULT_HOTPOT_DIR / "valid_title_sentence.json"
    paragraph_store_path: Path = DEFAULT_HOTPOT_DIR / "vector_stores" / "valid_title_sentence"
    sentence_store_path: Path = DEFAULT_HOTPOT_DIR / "vector_stores" / "single_sentence"
    output_dir: Path = PROJECT_ROOT / "new_experiments" / "results"

    sample_size: int = 10
    random_seed: int = 42
    k1: int = 10
    k2: int = 20
    k3: int = 7
    hmax: int = 2
    context_budget: int = 3600
    complexity_threshold: float = 0.80
    parent_threshold: float = 0.60
    fragmentation_threshold: float = 0.65
    max_graph_neighbors: int = 10
    max_context_units: int = 20

    run_generation: bool = True
    run_judge: bool = True
    use_api_reranker: bool = False
    require_neo4j: bool = True
    complexity_stratified_sampling: bool = True
    
    max_workers: int = 1
    checkpoint_interval: int = 10
    
    retrieval_cache_dir: str = ""
    skip_retrieval: bool = False


@dataclass
class EvidenceUnit:
    """一个检索到的证据单元，可以是句子、段落或摘要粒度。"""

    id: str
    title: str
    content: str
    score: float = 0.0
    source: str = "unknown"
    granularity: str = "sentence"
    is_sentence_level: bool = False
    sentence_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return estimate_tokens(self.content)


@dataclass
class QueryComplexity:
    score: float
    entity_count: int
    relation_constraint_count: int
    multi_hop_indicator: bool
    text_length: int
    detail: Dict[str, float] = field(default_factory=dict)


@dataclass
class EvidenceStatus:
    score: float
    concentration: float
    fragmentation: float
    complementarity: float
    short_sentence_ratio: float
    shared_parent_ratio: float
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetrics:
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    map_score: float = 0.0
    avg_len: float = 0.0
    time_ms: float = 0.0
    expanded_nodes: float = 0.0
    storage_mb: Optional[float] = None


@dataclass
class MethodResult:
    units: List[EvidenceUnit]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RerankSearchResult:
    """现有重排序器实现接受的最小适配器。"""

    doc_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryComplexityScorer:
    """基于规则的实现。"""

    QUESTION_WORDS = {
        "a",
        "an",
        "and",
        "are",
        "did",
        "do",
        "does",
        "for",
        "from",
        "in",
        "is",
        "of",
        "the",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
    }

    MULTI_HOP_PATTERNS = [
        r"\b(same|different|both|either|neither)\b",
        r"\b(compare|comparison|versus|vs\.?|difference|similar)\b",
        r"\b(before|after|during|while|until|since)\b",
        r"\b(author|director|founder|producer|creator).*\b(born|birth|nationality|country|city)\b",
        r"\b(who|what|which|where|when)\b.*\b(who|what|which|where|when)\b",
    ]

    RELATION_PATTERNS = [
        r"\b(same|different)\s+(nationality|country|state|city|language|genre)\b",
        r"\b(older|younger|earlier|later|larger|smaller|more|less|fewer|greater)\b",
        r"\b(belong|owned|founded|created|established|directed|written|produced|born)\b",
        r"\b(nationality|country|birthplace|occupation|genre|release|location)\b",
    ]

    def compute(self, query: str) -> QueryComplexity:
        entities = self._extract_entities(query)
        relation_count = self._count_relation_constraints(query)
        multi_hop = self._detect_multi_hop(query)
        text_length = len(query)

        entity_score = min(len(entities) / 2.0, 1.0)
        relation_score = min(relation_count / 1.0, 1.0)
        multi_hop_score = 1.0 if multi_hop else 0.0
        length_score = min(text_length / 120.0, 1.0)

        score = (
            0.35 * entity_score
            + 0.25 * relation_score
            + 0.25 * multi_hop_score
            + 0.15 * length_score
        )

        return QueryComplexity(
            score=round(min(score, 1.0), 4),
            entity_count=len(entities),
            relation_constraint_count=relation_count,
            multi_hop_indicator=multi_hop,
            text_length=text_length,
            detail={
                "entity_score": round(entity_score, 4),
                "relation_score": round(relation_score, 4),
                "multi_hop_score": round(multi_hop_score, 4),
                "length_score": round(length_score, 4),
            },
        )

    def _extract_entities(self, query: str) -> Set[str]:
        entities: Set[str] = set()

        for quoted in re.findall(r'"([^"]+)"|\'([^\']+)\'', query):
            item = quoted[0] or quoted[1]
            if item:
                entities.add(item.strip().lower())

        for match in re.findall(r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+)*\b", query):
            cleaned = match.strip()
            if cleaned.lower() not in self.QUESTION_WORDS and len(cleaned) > 1:
                entities.add(cleaned.lower())

        return entities

    def _count_relation_constraints(self, query: str) -> int:
        return sum(1 for pattern in self.RELATION_PATTERNS if re.search(pattern, query, re.IGNORECASE))

    def _detect_multi_hop(self, query: str) -> bool:
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in self.MULTI_HOP_PATTERNS)


COMPLEXITY_LEVELS = ("low", "medium", "high")


def complexity_level_for_score(score: float, complexity_threshold: float = 0.80) -> str:
    if score < 0.45:
        return "low"
    if score < complexity_threshold:
        return "medium"
    return "high"


def _balanced_complexity_targets(sample_size: int) -> Dict[str, int]:
    sample_size = max(0, int(sample_size or 0))
    base, remainder = divmod(sample_size, len(COMPLEXITY_LEVELS))
    return {
        level: base + (1 if idx < remainder else 0)
        for idx, level in enumerate(COMPLEXITY_LEVELS)
    }


def _assign_ranked_complexity_levels(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        scored["_complexity_level"] = []
        return scored

    ranked = scored.sort_values(["_complexity_score", "id"], kind="mergesort").copy()
    n = len(ranked)
    base, remainder = divmod(n, len(COMPLEXITY_LEVELS))
    labels: List[str] = []
    for idx, level in enumerate(COMPLEXITY_LEVELS):
        labels.extend([level] * (base + (1 if idx < remainder else 0)))
    ranked["_complexity_level"] = labels[:n]
    return ranked.sort_index()


def sample_hotpotqa_by_complexity(
    data: pd.DataFrame,
    sample_size: int,
    scorer: QueryComplexityScorer,
    random_seed: int,
    complexity_threshold: float = 0.80,
) -> pd.DataFrame:
    """Sample HotpotQA rows by low/medium/high query complexity."""
    if not sample_size or sample_size >= len(data):
        return data.reset_index(drop=True)

    scored = data.copy()
    scored["_complexity_score"] = [
        scorer.compute(str(question)).score
        for question in scored["question"]
    ]
    scored = _assign_ranked_complexity_levels(scored)

    targets = _balanced_complexity_targets(sample_size)
    selected_parts: List[pd.DataFrame] = []
    selected_indexes: Set[Any] = set()
    for offset, level in enumerate(COMPLEXITY_LEVELS):
        target = targets[level]
        bucket = scored[scored["_complexity_level"] == level]
        if target <= 0 or bucket.empty:
            continue
        take = min(target, len(bucket))
        part = bucket.sample(n=take, random_state=random_seed + offset)
        selected_parts.append(part)
        selected_indexes.update(part.index.tolist())

    selected_count = sum(len(part) for part in selected_parts)
    remaining = sample_size - selected_count
    if remaining > 0:
        rest = scored.drop(index=list(selected_indexes), errors="ignore")
        if not rest.empty:
            selected_parts.append(rest.sample(n=min(remaining, len(rest)), random_state=random_seed + 99))

    if not selected_parts:
        return scored.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    sampled = pd.concat(selected_parts, axis=0)
    return sampled.sample(frac=1.0, random_state=random_seed + 123).reset_index(drop=True)


def estimate_tokens(text: str) -> int:
    """用于上下文预算核算的确定性token估算。"""

    if not text:
        return 0
    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return max(1, int(chinese_chars * 1.5 + other_chars / 4.0))


def compute_retrieval_metrics(
    retrieved_titles: Sequence[str],
    relevant_titles: Set[str],
    avg_context_len: float,
    latency_ms: float,
    expanded_nodes: float = 0.0,
    storage_mb: Optional[float] = None,
) -> ExperimentMetrics:
    """计算标题级别的Recall、Precision、MRR、NDCG和MAP。"""

    relevant = set(relevant_titles)
    retrieved = list(dict.fromkeys(title for title in retrieved_titles if title))
    if not relevant:
        return ExperimentMetrics(avg_len=avg_context_len, time_ms=latency_ms, expanded_nodes=expanded_nodes, storage_mb=storage_mb)

    hits = [1 if title in relevant else 0 for title in retrieved]
    hit_count = sum(hits)
    recall = hit_count / len(relevant)
    precision = hit_count / len(retrieved) if retrieved else 0.0

    mrr = 0.0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / idx
            break

    dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(hits))
    ideal_hits = [1] * min(len(relevant), len(retrieved))
    ideal_dcg = sum(hit / math.log2(idx + 2) for idx, hit in enumerate(ideal_hits))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0

    precisions = []
    running_hits = 0
    for idx, hit in enumerate(hits, start=1):
        if hit:
            running_hits += 1
            precisions.append(running_hits / idx)
    map_score = sum(precisions) / len(relevant) if relevant else 0.0

    return ExperimentMetrics(
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=ndcg,
        map_score=map_score,
        avg_len=avg_context_len,
        time_ms=latency_ms,
        expanded_nodes=expanded_nodes,
        storage_mb=storage_mb,
    )


def compute_support_fact_metrics(
    retrieved_units: Sequence[EvidenceUnit],
    relevant_facts: Set[Tuple[str, str]],
    avg_context_len: float,
    latency_ms: float,
    expanded_nodes: float = 0.0,
    storage_mb: Optional[float] = None,
) -> ExperimentMetrics:
    relevant = {(str(title), str(sent_id)) for title, sent_id in relevant_facts if str(title)}
    if not relevant:
        return compute_retrieval_metrics(
            retrieved_titles=[unit.title for unit in retrieved_units],
            relevant_titles=set(),
            avg_context_len=avg_context_len,
            latency_ms=latency_ms,
            expanded_nodes=expanded_nodes,
            storage_mb=storage_mb,
        )

    seen_units: Set[Tuple[str, str, str]] = set()
    seen_facts: Set[Tuple[str, str]] = set()
    unit_gains: List[int] = []
    for unit in retrieved_units:
        unit_key = (unit.title, unit.sentence_id, unit.content)
        if unit_key in seen_units:
            continue
        seen_units.add(unit_key)

        if unit.is_sentence_level and unit.sentence_id:
            covered = {(unit.title, str(unit.sentence_id))} & relevant
        else:
            covered = {fact for fact in relevant if fact[0] == unit.title}
        new_facts = covered - seen_facts
        unit_gains.append(len(new_facts))
        seen_facts.update(new_facts)

    hit_count = len(seen_facts)
    retrieved_count = len(unit_gains)
    recall = hit_count / len(relevant)
    precision = min(1.0, hit_count / retrieved_count) if retrieved_count else 0.0

    mrr = 0.0
    for idx, gain in enumerate(unit_gains, start=1):
        if gain:
            mrr = 1.0 / idx
            break

    dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(unit_gains))
    ideal_gains = [1] * min(len(relevant), retrieved_count)
    ideal_dcg = sum(gain / math.log2(idx + 2) for idx, gain in enumerate(ideal_gains))
    ndcg = min(1.0, dcg / ideal_dcg) if ideal_dcg else 0.0

    precisions = []
    running_hits = 0
    for idx, gain in enumerate(unit_gains, start=1):
        if gain:
            running_hits += gain
            precisions.extend([running_hits / idx] * gain)
    map_score = min(1.0, sum(precisions) / len(relevant)) if relevant else 0.0

    return ExperimentMetrics(
        recall=recall,
        precision=precision,
        mrr=mrr,
        ndcg=ndcg,
        map_score=map_score,
        avg_len=avg_context_len,
        time_ms=latency_ms,
        expanded_nodes=expanded_nodes,
        storage_mb=storage_mb,
    )


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    return sum(nums) / len(nums) if nums else None


def ci95(values: Iterable[Optional[float]]) -> Optional[float]:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not math.isnan(float(v))]
    if len(nums) < 2:
        return None
    return 1.96 * statistics.stdev(nums) / math.sqrt(len(nums))


def round2(value: Optional[float]) -> Optional[float]:
    """保留2位小数"""
    if value is None or not isinstance(value, (int, float)) or math.isnan(float(value)):
        return None
    return round(float(value), 2)


def unique_by_title(units: Iterable[EvidenceUnit], keep_content_distinct: bool = False) -> List[EvidenceUnit]:
    seen: Set[Tuple[str, str]] = set()
    out: List[EvidenceUnit] = []
    for unit in units:
        key = (unit.title, unit.content if keep_content_distinct else "")
        if unit.title and key not in seen:
            seen.add(key)
            out.append(unit)
    return out


def context_len(units: Sequence[EvidenceUnit]) -> int:
    return sum(unit.token_count for unit in units)


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total / (1024 * 1024)


class PaperExperimentRunner:
    """运行所有实验。"""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self.complexity_scorer = QueryComplexityScorer()
        self.generation_metrics = None

        self.embedding_client: Optional[Any] = None
        self.llm_client: Optional[Any] = None
        self.reranker: Optional[Any] = None
        self.graph_store: Optional[Any] = None
        self.sentence_store: Optional[Any] = None
        self.paragraph_store: Optional[Any] = None
        self.test_data: Optional[pd.DataFrame] = None

        self.documents: List[Dict[str, Any]] = []
        self.title_to_content: Dict[str, str] = {}
        
        self._rerank_semaphore = threading.Semaphore(5)
        self._rerank_cache: Dict[str, List[EvidenceUnit]] = {}
        self._graph_unit_cache: Dict[str, List[EvidenceUnit]] = {}
        self._graph_retrieve_cache: Dict[str, List[EvidenceUnit]] = {}
        self._graph_embedding_cache: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # 资源加载
    # ------------------------------------------------------------------
    def load_resources(self) -> None:
        logger.info("Loading HotpotQA resources for paper experiments")
        self._load_documents()
        self._load_test_data()
        self._load_vector_stores()
        self._load_embedding_client()
        self._load_graph_store()
        self._load_optional_clients()

    def _load_documents(self) -> None:
        if self.config.require_neo4j:
            self.documents = []
            self.title_to_content = {}
            logger.info("Neo4j-required run: online evidence will be read from Neo4j, not JSON documents")
            return

        with open(self.config.documents_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        self.title_to_content = {
            doc.get("title", ""): doc.get("sentence_total", doc.get("content", ""))
            for doc in self.documents
            if doc.get("title")
        }
        logger.info("Loaded %s title-level documents", len(self.title_to_content))

    def _load_test_data(self) -> None:
        data = pd.read_parquet(self.config.test_data_path)
        if self.config.sample_size and self.config.sample_size < len(data):
            if self.config.complexity_stratified_sampling:
                data = sample_hotpotqa_by_complexity(
                    data,
                    sample_size=self.config.sample_size,
                    scorer=self.complexity_scorer,
                    random_seed=self.config.random_seed,
                    complexity_threshold=self.config.complexity_threshold,
                )
            else:
                data = data.sample(n=self.config.sample_size, random_state=self.config.random_seed)
        self.test_data = data.reset_index(drop=True)
        logger.info("Loaded %s HotpotQA samples", len(self.test_data))

    def _load_vector_stores(self) -> None:
        from src.storage.vector_store.faiss_store import FAISSVectorStore

        self.sentence_store = FAISSVectorStore()
        self.sentence_store.load(str(self.config.sentence_store_path))
        logger.info("Loaded sentence vector store: %s vectors", self.sentence_store.count())

        self.paragraph_store = FAISSVectorStore()
        self.paragraph_store.load(str(self.config.paragraph_store_path))
        logger.info("Loaded paragraph vector store: %s vectors", self.paragraph_store.count())

    def _load_embedding_client(self) -> None:
        from src.llms.embedding_client import EmbeddingClient

        self.embedding_client = EmbeddingClient()
        logger.info("Embedding client ready: dimension=%s", self.embedding_client.dimension)

    def _load_graph_store(self) -> None:
        try:
            from src.storage.graph_store.neo4j_store import Neo4jGraphStore

            self.graph_store = Neo4jGraphStore()
            node_count = self.graph_store.count_nodes()
            edge_count = self.graph_store.count_edges()
            logger.info("Neo4j ready: nodes=%s, edges=%s", node_count, edge_count)
        except Exception as exc:
            if self.config.require_neo4j:
                raise
            logger.warning("Neo4j unavailable; graph methods will degrade: %s", exc)
            self.graph_store = None

    def _load_optional_clients(self) -> None:
        if self.config.run_generation or self.config.run_judge:
            from src.llms.deepseek_client import DeepSeekClient

            self.llm_client = DeepSeekClient()

        try:
            from src.retrievers.reranker import create_reranker

            mode = "api" if self.config.use_api_reranker else "local"
            self.reranker = create_reranker(mode=mode)
        except Exception as exc:
            logger.warning("Reranker unavailable; using lexical rerank: %s", exc)
            self.reranker = None

    # ------------------------------------------------------------------
    # 核心检索原语
    # ------------------------------------------------------------------
    def vector_retrieve(self, query: str, store_name: str = "sentence", top_k: Optional[int] = None) -> List[EvidenceUnit]:
        if self.config.require_neo4j and self.graph_store is not None:
            return self.graph_semantic_retrieve(query, store_name, top_k or self.config.k1)

        store = self.sentence_store if store_name == "sentence" else self.paragraph_store
        if store is None or self.embedding_client is None:
            return []

        qv = self.embedding_client.embed(query)
        if len(qv.shape) == 2:
            qv = qv[0]
        results = store.search(qv, top_k=top_k or self.config.k1)

        units = []
        for result in results:
            title = result.metadata.extra.get("title", result.metadata.doc_id)
            sentence_id = result.metadata.extra.get("sentence_id", "")
            content = result.metadata.content
            units.append(
                EvidenceUnit(
                    id=result.id,
                    title=title,
                    content=content,
                    score=float(result.score),
                    source=f"vector_{store_name}",
                    granularity="sentence" if sentence_id else "paragraph",
                    is_sentence_level=bool(sentence_id),
                    sentence_id=sentence_id,
                    metadata=dict(result.metadata.extra),
                )
            )
        return units

    def graph_semantic_retrieve(self, query: str, store_name: str = "sentence", top_k: Optional[int] = None) -> List[EvidenceUnit]:
        if self.graph_store is None or self.embedding_client is None:
            return []

        limit = top_k or self.config.k1
        cache_key = f"{store_name}::{query}::{limit}"
        if cache_key in self._graph_retrieve_cache:
            return self._graph_retrieve_cache[cache_key]

        candidates = self.load_graph_evidence_units(store_name)
        if not candidates:
            return []

        candidate_embeddings = self.load_graph_evidence_embeddings(store_name)
        qv = self.embedding_client.embed(query)
        if len(qv.shape) == 2:
            qv = qv[0]
        scores = np.dot(candidate_embeddings, qv)
        scored: List[Tuple[float, EvidenceUnit]] = []
        for score, unit in zip(scores, candidates):
            scored.append(
                (
                    float(score),
                    EvidenceUnit(
                        id=unit.id,
                        title=unit.title,
                        content=unit.content,
                        score=float(score),
                        source=f"graph_semantic_{store_name}",
                        granularity=unit.granularity,
                        is_sentence_level=unit.is_sentence_level,
                        sentence_id=unit.sentence_id,
                        metadata={**unit.metadata, "section_source": "neo4j"},
                    ),
                )
            )
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [unit for _, unit in scored[:limit]]
        self._graph_retrieve_cache[cache_key] = result
        return result

    def load_graph_evidence_embeddings(self, store_name: str) -> np.ndarray:
        if store_name in self._graph_embedding_cache:
            return self._graph_embedding_cache[store_name]
        if self.embedding_client is None:
            return np.zeros((0, 0), dtype=np.float32)

        units = self.load_graph_evidence_units(store_name)
        if not units:
            embeddings = np.zeros((0, self.embedding_client.dimension), dtype=np.float32)
            self._graph_embedding_cache[store_name] = embeddings
            return embeddings

        vectors_by_index: Dict[int, np.ndarray] = {}
        missing_indexes: List[int] = []
        missing_texts: List[str] = []
        for idx, unit in enumerate(units):
            stored_vector = self._stored_graph_embedding(unit)
            if stored_vector is None:
                missing_indexes.append(idx)
                missing_texts.append(f"{unit.title}\n{unit.content}")
            else:
                vectors_by_index[idx] = stored_vector

        if missing_texts:
            encoded = self.embedding_client.embed(missing_texts)
            for idx, vector in zip(missing_indexes, encoded):
                vectors_by_index[idx] = np.asarray(vector, dtype=np.float32)

        dimension = next((len(vector) for vector in vectors_by_index.values() if len(vector)), self.embedding_client.dimension)
        embeddings = np.vstack(
            [
                vectors_by_index.get(idx, np.zeros(dimension, dtype=np.float32))
                for idx in range(len(units))
            ]
        ).astype(np.float32)
        self._graph_embedding_cache[store_name] = embeddings
        return embeddings

    @staticmethod
    def _stored_graph_embedding(unit: EvidenceUnit) -> Optional[np.ndarray]:
        vector = unit.metadata.get("graph_embedding")
        if vector is None:
            return None
        if hasattr(vector, "tolist"):
            vector = vector.tolist()
        if not isinstance(vector, list) or not vector:
            return None
        try:
            return np.asarray([float(value) for value in vector], dtype=np.float32)
        except (TypeError, ValueError):
            return None

    def load_graph_evidence_units(self, store_name: str) -> List[EvidenceUnit]:
        if store_name in self._graph_unit_cache:
            return self._graph_unit_cache[store_name]
        if self.graph_store is None:
            return []

        if store_name == "sentence":
            cypher = """
            MATCH (sec:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)-[:HAS_SENTENCE]->(sent:Sentence)
            RETURN sent.id AS id,
                   sec.title AS title,
                   sent.text AS content,
                   sent.sent_id AS sent_id,
                   p.id AS paragraph_id,
                   sent.embedding AS embedding
            ORDER BY sec.title, sent.position
            """
            rows = self.graph_store.query(cypher)
            units = [
                EvidenceUnit(
                    id=str(row.get("id") or f"sentence::{row.get('title')}::{row.get('sent_id')}"),
                    title=str(row.get("title") or ""),
                    content=str(row.get("content") or ""),
                    score=0.0,
                    source="graph_semantic_sentence",
                    granularity="sentence",
                    is_sentence_level=True,
                    sentence_id=str(row.get("sent_id") if row.get("sent_id") is not None else ""),
                    metadata={
                        "paragraph_id": str(row.get("paragraph_id") or ""),
                        "section_source": "neo4j",
                        "graph_embedding": row.get("embedding"),
                    },
                )
                for row in rows
                if row.get("title") and row.get("content")
            ]
        else:
            cypher = """
            MATCH (sec:Section)-[:HAS_PARAGRAPH]->(p:Paragraph)
            RETURN p.id AS id,
                   sec.title AS title,
                   p.text AS content,
                   p.position AS position,
                   p.embedding AS embedding
            ORDER BY sec.title, p.position
            """
            rows = self.graph_store.query(cypher)
            units = [
                EvidenceUnit(
                    id=str(row.get("id") or f"paragraph::{row.get('title')}"),
                    title=str(row.get("title") or ""),
                    content=str(row.get("content") or ""),
                    score=0.0,
                    source="graph_semantic_paragraph",
                    granularity="paragraph",
                    metadata={
                        "paragraph_id": str(row.get("id") or ""),
                        "position": row.get("position"),
                        "section_source": "neo4j",
                        "graph_embedding": row.get("embedding"),
                    },
                )
                for row in rows
                if row.get("title") and row.get("content")
            ]

        self._graph_unit_cache[store_name] = units
        return units

    def keyword_retrieve(self, query: str, top_k: Optional[int] = None) -> List[EvidenceUnit]:
        keywords = self.extract_keywords(query)
        if not keywords or self.graph_store is None:
            return []

        limit = top_k or self.config.k2
        cypher = """
        MATCH (n:Section)
        WHERE any(k IN $keywords WHERE toLower(n.title) CONTAINS toLower(k))
        RETURN n.title AS title, n.sentence_total AS content
        LIMIT $limit
        """
        try:
            rows = self.graph_store.query(cypher, {"keywords": keywords, "limit": limit})
        except Exception as exc:
            logger.warning("Keyword graph retrieval failed: %s", exc)
            return []

        units = []
        for row in rows:
            title = str(row.get("title", "")).strip('"')
            content = str(row.get("content", "") or self.title_to_content.get(title, ""))
            if title and content:
                units.append(
                    EvidenceUnit(
                        id=f"kw::{title}",
                        title=title,
                        content=content,
                        score=0.65,
                        source="keyword_graph",
                        granularity="paragraph",
                    )
                )
        return unique_by_title(units)

    def extract_keywords(self, query: str) -> List[str]:
        entities = sorted(self.complexity_scorer._extract_entities(query), key=len, reverse=True)
        keywords = [entity.strip() for entity in entities if entity.strip()]
        if keywords:
            return keywords[:10]

        stop = QueryComplexityScorer.QUESTION_WORDS | {"with", "that", "this", "into", "have", "has"}
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", query) if t.lower() not in stop and len(t) > 3]
        return list(dict.fromkeys(tokens))[:10]

    def rerank_units(self, query: str, units: Sequence[EvidenceUnit], top_k: Optional[int] = None) -> List[EvidenceUnit]:
        if not units:
            return []
        limit = top_k or self.config.k3
        
        cache_key = f"{query}::{','.join(u.id for u in units)}::{limit}"
        if cache_key in self._rerank_cache:
            return self._rerank_cache[cache_key]

        if self.reranker is not None:
            try:
                with self._rerank_semaphore:
                    search_results = [
                        RerankSearchResult(doc_id=u.id, content=u.content, score=u.score, metadata={"unit": u})
                        for u in units
                    ]
                    reranked = self.reranker.rerank(query, search_results, top_k=limit)
                result = [r.metadata["unit"] for r in reranked]
                self._rerank_cache[cache_key] = result
                return result
            except Exception as exc:
                logger.warning("API rerank failed; using lexical rerank: %s", exc)

        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))
        scored: List[Tuple[float, EvidenceUnit]] = []
        for unit in units:
            content_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", unit.content))
            title_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", unit.title))
            overlap = len(query_terms & (content_terms | title_terms)) / max(len(query_terms), 1)
            score = 0.7 * unit.score + 0.3 * overlap
            unit.score = score
            scored.append((score, unit))
        scored.sort(key=lambda item: item[0], reverse=True)
        result = [unit for _, unit in scored[:limit]]
        self._rerank_cache[cache_key] = result
        return result

    def parent_map(self, units: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
        mapped: Dict[Tuple[str, str, str], EvidenceUnit] = {}
        for unit in units:
            graph_parent = self.fetch_parent_from_graph(unit.title)
            if graph_parent:
                full = graph_parent.get("content")
            elif self.config.require_neo4j:
                full = ""
            else:
                full = self.title_to_content.get(unit.title)
            if full and (unit.is_sentence_level or len(unit.content) < len(full)):
                key = ("parent", unit.title, str(graph_parent.get("paragraph_id") if graph_parent else unit.title))
                if key not in mapped:
                    mapped[key] = EvidenceUnit(
                        id=f"parent::{unit.title}",
                        title=unit.title,
                        content=full,
                        score=unit.score,
                        source=f"{unit.source}+parent",
                        granularity="paragraph",
                        metadata={
                            "parent_of": unit.id,
                            "paragraph_id": graph_parent.get("paragraph_id") if graph_parent else "",
                            "section_source": "neo4j" if graph_parent else "json_fallback",
                            "trigger_unit_ids": [unit.id],
                            "trigger_sentence_ids": [unit.sentence_id] if unit.sentence_id else [],
                        },
                    )
                else:
                    parent = mapped[key]
                    parent.score = max(parent.score, unit.score)
                    parent.metadata.setdefault("trigger_unit_ids", []).append(unit.id)
                    if unit.sentence_id:
                        parent.metadata.setdefault("trigger_sentence_ids", []).append(unit.sentence_id)
            else:
                mapped[("unit", unit.title, unit.content)] = unit
        return list(mapped.values())

    def parent_context_pool(self, units: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
        parents = [
            unit
            for unit in self.parent_map(units)
            if unit.metadata.get("trigger_unit_ids") and unit.granularity in {"paragraph", "section"}
        ]
        return list(units) + parents

    def fetch_parent_from_graph(self, title: str) -> Optional[Dict[str, Any]]:
        if self.graph_store is None or not title:
            return None

        cypher = """
        MATCH (s:Section {title: $title})
        OPTIONAL MATCH (s)-[:HAS_PARAGRAPH]->(p:Paragraph)
        RETURN s.title AS title,
               coalesce(p.text, s.sentence_total) AS content,
               p.id AS paragraph_id
        ORDER BY p.position
        LIMIT 1
        """
        try:
            rows = self.graph_store.query(cypher, {"title": title})
        except Exception as exc:
            logger.warning("Parent lookup from graph failed for %s: %s", title, exc)
            return None
        if not rows:
            return None
        content = str(rows[0].get("content") or "")
        if not content:
            return None
        return {
            "title": str(rows[0].get("title") or title),
            "content": content,
            "paragraph_id": str(rows[0].get("paragraph_id") or ""),
        }

    def graph_expand(self, seed_titles: Sequence[str], hops: int = 1, limit_per_seed: Optional[int] = None) -> List[EvidenceUnit]:
        if self.graph_store is None:
            return []

        hops = max(1, min(int(hops), self.config.hmax))
        limit = limit_per_seed or self.config.max_graph_neighbors
        max_degree = 500
        expanded: List[EvidenceUnit] = []
        seen = set(seed_titles)

        for title in seed_titles:
            cypher = f"""
            MATCH (start:Section {{title: $title}})
            MATCH (start)-[:SEMANTIC_LINKS]->(first:Entity)
            MATCH p = (first)-[:RELATED*0..{hops}]-(last:Entity)
            MATCH (last)<-[:SEMANTIC_LINKS]-(n:Section)
            WHERE n <> start
              AND ALL(x IN nodes(p) WHERE COUNT {{ (x)--() }} <= $max_degree)
            RETURN DISTINCT n.title AS title, n.sentence_total AS content
            LIMIT $limit
            """
            try:
                rows = self.graph_store.query(
                    cypher,
                    {"title": title, "max_degree": max_degree, "limit": limit},
                )
            except Exception as exc:
                logger.warning("Graph expansion failed for %s: %s", title, exc)
                continue

            for row in rows:
                new_title = str(row.get("title", "")).strip('"')
                content = str(row.get("content", "") or self.title_to_content.get(new_title, ""))
                if new_title and content and new_title not in seen:
                    seen.add(new_title)
                    expanded.append(
                        EvidenceUnit(
                            id=f"graph::{new_title}",
                            title=new_title,
                            content=content,
                            score=0.55,
                            source=f"graph_{hops}hop",
                            granularity="paragraph",
                        )
                    )
        return expanded

    def add_summary_evidence(self, seed_titles: Sequence[str], query: str = "") -> List[EvidenceUnit]:
        """提取关键句子作为摘要证据
        
        Args:
            seed_titles: 种子标题列表
            query: 查询文本，用于提取相关句子
        """
        summaries = []
        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query)) if query else set()
        
        for title in seed_titles:
            graph_parent = self.fetch_parent_from_graph(title)
            if graph_parent:
                content = graph_parent.get("content", "")
            elif self.config.require_neo4j:
                content = ""
            else:
                content = self.title_to_content.get(title, "")
            if not content:
                continue
            
            if query_terms:
                sentences = re.split(r'(?<=[.!?])\s+', content)
                scored_sentences = []
                for sent in sentences:
                    sent_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", sent))
                    overlap = len(query_terms & sent_terms)
                    if overlap > 0:
                        scored_sentences.append((overlap, len(sent), sent))
                
                scored_sentences.sort(key=lambda x: (-x[0], x[1]))
                summary_sentences = [sent for _, _, sent in scored_sentences[:3]]
                summary = ' '.join(summary_sentences) if summary_sentences else content[:600]
            else:
                summary = content[:600]
            
            summary = summary[:900]
            
            summaries.append(
                EvidenceUnit(
                    id=f"summary::{title}",
                    title=title,
                    content=summary,
                    score=0.30,
                    source="summary",
                    granularity="summary",
                )
            )
        return summaries

    def select_with_budget(self, query: str, units: Sequence[EvidenceUnit], max_units: Optional[int] = None) -> List[EvidenceUnit]:
        selected: List[EvidenceUnit] = []
        selected_terms: Set[str] = set()
        budget = self.config.context_budget
        max_units = max_units or self.config.max_context_units
        total_tokens = 0

        candidates = unique_by_title(units, keep_content_distinct=True)
        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))

        remaining = []
        for index, unit in enumerate(candidates):
            terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", f"{unit.title} {unit.content}"))
            evidence_terms = terms & query_terms if query_terms else terms
            remaining.append((index, unit, evidence_terms))

        while remaining and len(selected) < max_units:
            scored = []
            for index, unit, evidence_terms in remaining:
                relevance = len(evidence_terms) / max(len(query_terms), 1) if query_terms else 0.0
                if not selected_terms:
                    completion = 1.0 if evidence_terms else 0.0
                else:
                    completion = len(evidence_terms - selected_terms) / max(len(evidence_terms), 1) if evidence_terms else 0.0
                cost = unit.token_count / max(budget, 1)
                value = 0.55 * relevance + 0.40 * completion - 0.15 * cost
                scored.append((value, unit.score, -index, index, unit, evidence_terms))

            scored.sort(reverse=True)
            _, _, _, selected_index, unit, evidence_terms = scored[0]
            if len(selected) >= max_units:
                break
            if total_tokens + unit.token_count > budget and selected:
                remaining = [item for item in remaining if item[0] != selected_index]
                continue
            selected.append(unit)
            selected_terms.update(evidence_terms)
            total_tokens += unit.token_count
            remaining = [item for item in remaining if item[0] != selected_index]

        return self.assemble_context(selected)

    def assemble_context(self, units: Sequence[EvidenceUnit]) -> List[EvidenceUnit]:
        def priority(unit: EvidenceUnit) -> Tuple[int, float]:
            source = unit.source.lower()
            if unit.granularity == "summary" or source == "summary":
                group = 2
            elif unit.id.startswith("graph::") or (source.startswith("graph_") and "semantic" not in source):
                group = 1
            else:
                group = 0
            return (group, -unit.score)

        return sorted(units, key=priority)

    # ------------------------------------------------------------------
    # 方法实现
    # ------------------------------------------------------------------
    def retrieve_semantic_rag(self, query: str) -> MethodResult:
        start = time.perf_counter()
        units = unique_by_title(self.vector_retrieve(query, "sentence", self.config.k1))[: self.config.k3]
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=0, route="semantic"))

    def retrieve_rerank_rag(self, query: str) -> MethodResult:
        start = time.perf_counter()
        candidates = self.vector_retrieve(query, "sentence", self.config.k1)
        units = unique_by_title(self.rerank_units(query, candidates, self.config.k3))
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=0, route="rerank"))

    def retrieve_graphrag(self, query: str) -> MethodResult:
        start = time.perf_counter()
        seeds = unique_by_title(self.vector_retrieve(query, "sentence", self.config.k1))[: self.config.k3]
        expanded = self.graph_expand([u.title for u in seeds], hops=self.config.hmax)
        units = self.select_with_budget(query, list(seeds) + expanded)
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=len(expanded), route="graphrag"))

    def retrieve_kg_rag(self, query: str) -> MethodResult:
        start = time.perf_counter()
        vector_units = self.vector_retrieve(query, "sentence", self.config.k1)
        keyword_units = self.keyword_retrieve(query, self.config.k2)
        seeds = self.rerank_units(query, vector_units + keyword_units, self.config.k3)
        expanded = self.graph_expand([u.title for u in seeds], hops=1)
        units = self.select_with_budget(query, seeds + expanded, max_units=self.config.k3 + self.config.max_graph_neighbors)
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=len(expanded), route="kg_rag"))

    def retrieve_macrag(self, query: str) -> MethodResult:
        start = time.perf_counter()
        sent = self.vector_retrieve(query, "sentence", self.config.k1)
        para = self.vector_retrieve(query, "paragraph", max(1, self.config.k1 // 2))
        units = self.rerank_units(query, sent + para, self.config.k3)
        units = self.select_with_budget(query, units, max_units=self.config.k3)
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=0, route="macrag"))

    def retrieve_adaptive(
        self,
        query: str,
        enable_graph_expansion: bool = True,
        enable_parent: bool = True,
        enable_summary: bool = True,
        forced_hops: Optional[int] = None,
        force_parent: bool = False,
        fine_only: bool = False,
    ) -> MethodResult:
        start = time.perf_counter()
        initial = self.vector_retrieve(query, "sentence", self.config.k1) + self.keyword_retrieve(query, self.config.k2)
        initial = unique_by_title(initial, keep_content_distinct=True)
        reranked = self.rerank_units(query, initial, self.config.k3)

        if fine_only:
            units = unique_by_title(reranked)[: self.config.k3]
            return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=0, route="fine_only"))

        if force_parent and enable_parent:
            units = self.parent_map(reranked)
            return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=0, route="parent_all"))

        route, route_detail = self.choose_adaptive_route(query, reranked)
        selected: List[EvidenceUnit]
        expanded: List[EvidenceUnit] = []

        if route == "fine_grained":
            selected = unique_by_title(reranked)[: self.config.k3]
        elif route == "local_parent" and enable_parent:
            selected = self.select_with_budget(query, self.parent_context_pool(reranked))
        else:
            parent_units = [
                unit
                for unit in (self.parent_map(reranked) if enable_parent else [])
                if unit.metadata.get("trigger_unit_ids")
            ]
            summary_seed_units = parent_units or list(reranked)
            hops = forced_hops or self._dynamic_hops(route_detail)
            if enable_graph_expansion:
                expanded = self.graph_expand([u.title for u in reranked], hops=hops)
            
            all_candidates = list(reranked) + parent_units + expanded
            current_tokens = sum(estimate_tokens(u.content) for u in parent_units + expanded)
            context_deficit = self.config.context_budget * 0.5 - current_tokens
            
            if enable_summary and context_deficit > 0:
                summaries = self.add_summary_evidence([u.title for u in summary_seed_units], query)
            else:
                summaries = []
            
            selected = self.select_with_budget(query, all_candidates + summaries)

        stats = self._stats(start, selected, expanded_nodes=len(expanded), route=route)
        stats["route_detail"] = route_detail
        return MethodResult(units=selected, stats=stats)

    def retrieve_fixed_graph(self, query: str, hops: int) -> MethodResult:
        start = time.perf_counter()
        initial = self.vector_retrieve(query, "sentence", self.config.k1) + self.keyword_retrieve(query, self.config.k2)
        initial = unique_by_title(initial, keep_content_distinct=True)
        seeds = self.rerank_units(query, initial, self.config.k3)
        expanded = self.graph_expand([u.title for u in seeds], hops=hops)
        units = self.select_with_budget(query, self.parent_context_pool(seeds) + expanded)
        return MethodResult(units=units, stats=self._stats(start, units, expanded_nodes=len(expanded), route=f"fixed_{hops}hop"))

    def choose_adaptive_route(self, query: str, candidates: Sequence[EvidenceUnit]) -> Tuple[str, Dict[str, Any]]:
        complexity = self.complexity_scorer.compute(query)
        evidence_status = self.evaluate_evidence_status(candidates, query)

        if (
            complexity.multi_hop_indicator
            or complexity.score >= self.config.complexity_threshold
            or evidence_status.fragmentation >= self.config.fragmentation_threshold
        ):
            route = "graph_expansion"
        elif evidence_status.score >= self.config.parent_threshold and evidence_status.short_sentence_ratio < 0.4:
            route = "fine_grained"
        else:
            route = "local_parent"

        return route, {
            "complexity_score": complexity.score,
            "entity_count": complexity.entity_count,
            "relation_constraint_count": complexity.relation_constraint_count,
            "multi_hop_indicator": complexity.multi_hop_indicator,
            "evidence_score": evidence_status.score,
            "concentration": evidence_status.concentration,
            "fragmentation": evidence_status.fragmentation,
            "complementarity": evidence_status.complementarity,
            "short_sentence_ratio": evidence_status.short_sentence_ratio,
            "shared_parent_ratio": evidence_status.shared_parent_ratio,
            "route": route,
        }

    def evaluate_evidence_status(self, candidates: Sequence[EvidenceUnit], query: str) -> EvidenceStatus:
        if not candidates:
            return EvidenceStatus(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, {"reason": "empty"})

        titles = [u.title for u in candidates if u.title]
        unique_titles = set(titles)
        max_same_title = max((titles.count(t) for t in unique_titles), default=0)
        concentration = max_same_title / len(titles) if titles else 0.0
        fragmentation = (len(unique_titles) - 1) / max(len(titles), 1)

        query_terms = set(t.lower() for t in re.findall(r"[A-Za-z0-9]+", query))
        content = " ".join(f"{u.title} {u.content}" for u in candidates).lower()
        complementarity = len([t for t in query_terms if t in content]) / max(len(query_terms), 1)

        short_sentence_ratio = sum(1 for u in candidates if u.is_sentence_level and u.token_count < 18) / len(candidates)
        shared_parent_ratio = 1.0 - len(unique_titles) / max(len(titles), 1)
        evidence_score = 0.40 * concentration + 0.35 * complementarity + 0.25 * (1.0 - fragmentation)

        return EvidenceStatus(
            score=round(evidence_score, 4),
            concentration=round(concentration, 4),
            fragmentation=round(fragmentation, 4),
            complementarity=round(complementarity, 4),
            short_sentence_ratio=round(short_sentence_ratio, 4),
            shared_parent_ratio=round(shared_parent_ratio, 4),
            detail={"unique_titles": len(unique_titles), "candidate_count": len(candidates)},
        )

    def _dynamic_hops(self, route_detail: Dict[str, Any]) -> int:
        high_complexity = route_detail.get("complexity_score", 0.0) >= self.config.complexity_threshold
        weak_complement = route_detail.get("complementarity", 0.0) < self.config.complexity_threshold
        return self.config.hmax if high_complexity and weak_complement else 1

    def _stats(self, start: float, units: Sequence[EvidenceUnit], expanded_nodes: int, route: str) -> Dict[str, Any]:
        return {
            "time_ms": (time.perf_counter() - start) * 1000,
            "avg_len": context_len(units),
            "expanded_nodes": expanded_nodes,
            "route": route,
        }

    def _llm_retry_count(self) -> int:
        if self.llm_client is None:
            return 1
        try:
            return max(1, int(getattr(self.llm_client, "max_retries", 3) or 1))
        except (TypeError, ValueError):
            return 3

    def _llm_retry_delay(self) -> float:
        if self.llm_client is None:
            return 0.0
        try:
            return max(0.0, float(getattr(self.llm_client, "retry_delay", 0.0) or 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _escalated_max_tokens(max_tokens: int, attempt: int) -> int:
        try:
            requested = max(1, int(max_tokens))
        except (TypeError, ValueError):
            requested = 256
        try:
            from src.utils.config import get_config

            cap = max(requested, int(get_config().llm_max_tokens))
        except Exception:
            cap = max(requested, 4096)
        return min(cap, requested * (2 ** attempt))

    def _generate_llm_response(
        self,
        messages: List[Any],
        temperature: float,
        max_tokens: int,
    ):
        from src.llms.base_client import LLMResponse

        last_response = None
        attempts = self._llm_retry_count()
        retry_delay = self._llm_retry_delay()
        for attempt in range(attempts):
            effective_max_tokens = self._escalated_max_tokens(max_tokens, attempt)
            response = self.llm_client.generate(
                messages,
                temperature=temperature,
                max_tokens=effective_max_tokens,
            )
            last_response = response
            if response.success and response.content and response.content.strip():
                return response

            logger.warning(
                "LLM returned empty/failed response (attempt %d/%d): %s",
                attempt + 1,
                attempts,
                getattr(response, "error", None) or "empty content",
            )
            if attempt < attempts - 1 and retry_delay > 0:
                time.sleep(retry_delay * (attempt + 1))

        return last_response or LLMResponse(content="", success=False, error="no llm response")

    # ------------------------------------------------------------------
    # 生成和判断
    # ------------------------------------------------------------------
    def generate_answer(self, question: str, units: Sequence[EvidenceUnit]) -> str:
        if not self.config.run_generation or self.llm_client is None:
            return ""

        evidence = "\n\n".join(
            f"[{idx}] {unit.title}: {unit.content}" for idx, unit in enumerate(units, start=1)
        )
        prompt = (
            "Answer the question using only the evidence below. "
            "If the evidence is insufficient, say that it cannot be determined.\n\n"
            f"Question: {question}\n\nEvidence:\n{evidence}\n\nAnswer:"
        )
        from src.llms.base_client import Message

        response = self._generate_llm_response(
            [Message(role="user", content=prompt)],
            temperature=0.0,
            max_tokens=256,
        )
        return response.content.strip() if response.success else ""

    def judge_answer(
        self,
        question: str,
        ground_truth: str,
        answer: str,
        units: Sequence[EvidenceUnit],
    ) -> Dict[str, Optional[float]]:
        base = {
            "correctness": None,
            "faithfulness": None,
            "answer_relevance": None,
            "context_relevance": None,
        }
        if not self.config.run_judge or self.llm_client is None or not answer:
            return base

        context = "\n\n".join(f"{u.title}: {u.content}" for u in units)[:5000]
        prompt = f"""You are a RAG evaluator. Score each metric from 0 to 1.

Question: {question}
Reference answer: {ground_truth}
Generated answer: {answer}

Retrieved context:
{context}

Return only JSON with keys correctness, faithfulness, answer_relevance, context_relevance.
"""
        from src.llms.base_client import Message

        messages = [Message(role="user", content=prompt)]
        parsed = None
        attempts = self._llm_retry_count()
        retry_delay = self._llm_retry_delay()
        for attempt in range(attempts):
            response = self.llm_client.generate(
                messages,
                temperature=0.0,
                max_tokens=self._escalated_max_tokens(512, attempt),
                response_format={"type": "json_object"},
            )
            if response.success and response.content and response.content.strip():
                parsed = self._parse_json_object(response.content)
                if parsed:
                    break

            logger.warning(
                "LLM judge returned failed/blank/unparsable response (attempt %d/%d): %s",
                attempt + 1,
                attempts,
                getattr(response, "error", None) or "unparsable JSON",
            )
            if attempt < attempts - 1 and retry_delay > 0:
                time.sleep(retry_delay * (attempt + 1))

        if not parsed:
            return base
        for key in base:
            try:
                base[key] = max(0.0, min(1.0, float(parsed.get(key))))
            except (TypeError, ValueError):
                base[key] = None
        return base

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------
    # 实验部分
    # ------------------------------------------------------------------
    def run_all(self) -> Dict[str, Any]:
        self.load_resources()
        assert self.test_data is not None

        run_dir = self._create_run_dir()
        rows_by_section: Dict[str, Any] = {}

        method_fns: Dict[str, Callable[[str], MethodResult]] = {
            "Semantic RAG": self.retrieve_semantic_rag,
            "+Rerank": self.retrieve_rerank_rag,
            "GraphRAG": self.retrieve_graphrag,
            "KG-RAG": self.retrieve_kg_rag,
            "MacRAG": self.retrieve_macrag,
            "Proposed": lambda q: self.retrieve_adaptive(q),
        }
        method_rows = self.evaluate_methods(method_fns, desc="主方法对比")
        rows_by_section["method_rows"] = method_rows
        method_summary = self.summarize_methods(method_rows)
        self.write_csv(run_dir / "table5_6_method_retrieval.csv", method_summary)
        self.write_csv(run_dir / "table7_semantic_records.csv", self.semantic_record_table(method_rows))

        fixed_fns: Dict[str, Callable[[str], MethodResult]] = {
            "Fine only": lambda q: self.retrieve_adaptive(q, fine_only=True),
            "Uniform parent": lambda q: self.retrieve_adaptive(q, force_parent=True),
            "Fixed 1-hop": lambda q: self.retrieve_fixed_graph(q, hops=1),
            "Fixed 2-hop": lambda q: self.retrieve_fixed_graph(q, hops=2),
            "Proposed": lambda q: self.retrieve_adaptive(q),
        }
        fixed_rows = self.evaluate_methods(fixed_fns, desc="固定尺度对比")
        rows_by_section["fixed_rows"] = fixed_rows
        self.write_csv(run_dir / "table8_9_fixed_scale.csv", self.summarize_methods(fixed_rows))

        ablation_fns: Dict[str, Callable[[str], MethodResult]] = {
            "Full model": lambda q: self.retrieve_adaptive(q),
            "No graph expansion": lambda q: self.retrieve_adaptive(q, enable_graph_expansion=False),
            "No selective parent": lambda q: self.retrieve_adaptive(q, enable_parent=False),
            "No summary evidence": lambda q: self.retrieve_adaptive(q, enable_summary=False),
        }
        ablation_rows = self.evaluate_methods(ablation_fns, desc="消融实验")
        rows_by_section["ablation_rows"] = ablation_rows
        self.write_csv(run_dir / "table10_ablation_retrieval.csv", self.summarize_methods(ablation_rows))
        self.write_csv(run_dir / "table11_ablation_semantic.csv", self.semantic_record_table(ablation_rows))

        complexity_rows = self.complexity_stratified_table(method_rows.get("Proposed", []))
        rows_by_section["complexity_rows"] = complexity_rows
        self.write_csv(run_dir / "table12_complexity_stratified.csv", complexity_rows)

        efficiency_rows = self.efficiency_table(method_summary)
        rows_by_section["efficiency_rows"] = efficiency_rows
        self.write_csv(run_dir / "table13_efficiency.csv", efficiency_rows)

        details_path = run_dir / "details.json"
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(rows_by_section), f, ensure_ascii=False, indent=2)

        config_path = run_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self._json_safe(asdict(self.config)), f, ensure_ascii=False, indent=2)

        logger.info("Paper experiments complete. Results: %s", run_dir)
        return {
            "run_dir": str(run_dir),
            "tables": {
                "method_retrieval": str(run_dir / "table5_6_method_retrieval.csv"),
                "semantic_records": str(run_dir / "table7_semantic_records.csv"),
                "fixed_scale": str(run_dir / "table8_9_fixed_scale.csv"),
                "ablation_retrieval": str(run_dir / "table10_ablation_retrieval.csv"),
                "ablation_semantic": str(run_dir / "table11_ablation_semantic.csv"),
                "complexity_stratified": str(run_dir / "table12_complexity_stratified.csv"),
                "efficiency": str(run_dir / "table13_efficiency.csv"),
                "details": str(details_path),
            },
        }

    def evaluate_methods(self, method_fns: Dict[str, Callable[[str], MethodResult]], desc: str = "评估方法") -> Dict[str, List[Dict[str, Any]]]:
        """评估所有方法（支持样本×方法双层并发）
        
        Args:
            method_fns: 方法名称到函数的映射
            desc: 进度条描述
        """
        assert self.test_data is not None
        rows_by_method: Dict[str, List[Dict[str, Any]]] = {name: [] for name in method_fns}
        rows_lock = threading.Lock()
        
        samples = list(self.test_data.iterrows())
        method_names = list(method_fns.keys())
        
        def retrieval_phase(idx: int, sample: pd.Series, method_name: str) -> Tuple[str, pd.Series, MethodResult, Set[str]]:
            """阶段1：检索（受重排序限制）"""
            fn = method_fns[method_name]
            question = str(sample.get("question", ""))
            relevant_titles = self.get_relevant_titles(sample)

            for attempt in range(3):
                try:
                    method_result = fn(question)
                    break
                except Exception as exc:
                    if attempt < 2:
                        time.sleep(1)
                        continue
                    logger.error("%s failed for sample %s: %s", method_name, sample.get("id"), exc)
                    method_result = MethodResult(units=[], stats={"error": str(exc), "time_ms": 0.0, "avg_len": 0.0, "expanded_nodes": 0})

            return method_name, sample, method_result, relevant_titles

        def generation_phase(method_name: str, sample: pd.Series, method_result: MethodResult, relevant_titles: Set[str]) -> Tuple[str, Dict[str, Any]]:
            """阶段2：生成和评估（不受限制）"""
            question = str(sample.get("question", ""))
            ground_truth = str(sample.get("answer", ""))
            relevant_facts = self.get_relevant_facts(sample)

            retrieved_titles = [u.title for u in method_result.units]
            metrics = compute_support_fact_metrics(
                retrieved_units=method_result.units,
                relevant_facts=relevant_facts,
                avg_context_len=method_result.stats.get("avg_len", context_len(method_result.units)),
                latency_ms=method_result.stats.get("time_ms", 0.0),
                expanded_nodes=method_result.stats.get("expanded_nodes", 0),
            )

            answer = self.generate_answer(question, method_result.units)
            semantic = self.judge_answer(question, ground_truth, answer, method_result.units)

            route_detail = method_result.stats.get("route_detail", {})
            complexity_score = route_detail.get("complexity_score", self.complexity_scorer.compute(question).score)
            result = {
                "id": sample.get("id"),
                "question": question,
                "answer": ground_truth,
                "type": sample.get("type"),
                "level": sample.get("level"),
                "relevant_titles": sorted(relevant_titles),
                "retrieved_titles": retrieved_titles,
                "retrieved_contexts": [u.content for u in method_result.units],
                "generated_answer": answer,
                "retrieval_metrics": metrics,
                "semantic_metrics": semantic,
                "stats": method_result.stats,
                "complexity_score": complexity_score,
                "complexity_level": sample.get("_complexity_level", complexity_level_for_score(float(complexity_score), self.config.complexity_threshold)),
                "route": method_result.stats.get("route", route_detail.get("route", "")),
            }
            
            return method_name, result
        
        cache_file = None
        if self.config.retrieval_cache_dir:
            cache_dir = Path(self.config.retrieval_cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / f"retrieval_{desc.replace(' ', '_')}.pkl"
        
        retrieval_results: List[Tuple[str, pd.Series, MethodResult, Set[str]]] = []
        generation_workers = self.config.max_workers
        
        if self.config.skip_retrieval and cache_file and cache_file.exists():
            logger.info("从缓存加载检索结果: %s", cache_file)
            import pickle
            with open(cache_file, "rb") as f:
                retrieval_results = pickle.load(f)
            logger.info("加载了 %d 条检索结果", len(retrieval_results))
        elif self.config.max_workers > 1:
            total_tasks = len(samples) * len(method_names)
            retrieval_workers = min(50, self.config.max_workers)
            
            pbar = tqdm(total=total_tasks, desc=f"{desc} - 检索阶段")
            
            retrieval_lock = threading.Lock()
            
            with ThreadPoolExecutor(max_workers=retrieval_workers) as retrieval_executor:
                futures = {}
                for idx, sample in samples:
                    for method_name in method_names:
                        future = retrieval_executor.submit(retrieval_phase, idx, sample, method_name)
                        futures[future] = (idx, method_name)
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        with retrieval_lock:
                            retrieval_results.append(result)
                        pbar.update(1)
                    except Exception as exc:
                        idx, method_name = futures[future]
                        logger.error("检索阶段异常 (idx=%s, method=%s): %s", idx, method_name, exc)
            
            pbar.close()
            
            if cache_file:
                logger.info("保存检索结果到缓存: %s", cache_file)
                import pickle
                with open(cache_file, "wb") as f:
                    pickle.dump(retrieval_results, f)
        else:
            for idx, sample in tqdm(samples, desc=f"{desc} - 检索阶段"):
                for method_name in method_names:
                    method_name_r, sample_r, method_result, relevant_titles = retrieval_phase(idx, sample, method_name)
                    retrieval_results.append((method_name_r, sample_r, method_result, relevant_titles))
        
        if not self.config.run_generation and not self.config.run_judge:
            logger.info("跳过生成阶段（run_generation=False, run_judge=False）")
            for method_name, sample, method_result, relevant_titles in retrieval_results:
                question = str(sample.get("question", ""))
                ground_truth = str(sample.get("answer", ""))
                relevant_facts = self.get_relevant_facts(sample)
                retrieved_titles = [u.title for u in method_result.units]
                metrics = compute_support_fact_metrics(
                    retrieved_units=method_result.units,
                    relevant_facts=relevant_facts,
                    avg_context_len=method_result.stats.get("avg_len", context_len(method_result.units)),
                    latency_ms=method_result.stats.get("time_ms", 0.0),
                    expanded_nodes=method_result.stats.get("expanded_nodes", 0),
                )
                route_detail = method_result.stats.get("route_detail", {})
                complexity_score = route_detail.get("complexity_score", self.complexity_scorer.compute(question).score)
                result = {
                    "id": sample.get("id"),
                    "question": question,
                    "answer": ground_truth,
                    "type": sample.get("type"),
                    "level": sample.get("level"),
                    "relevant_titles": sorted(relevant_titles),
                    "retrieved_titles": retrieved_titles,
                    "retrieved_contexts": [u.content for u in method_result.units],
                    "generated_answer": "",
                    "retrieval_metrics": metrics,
                    "semantic_metrics": {},
                    "stats": method_result.stats,
                    "complexity_score": complexity_score,
                    "complexity_level": sample.get("_complexity_level", complexity_level_for_score(float(complexity_score), self.config.complexity_threshold)),
                    "route": method_result.stats.get("route", route_detail.get("route", "")),
                }
                with rows_lock:
                    rows_by_method[method_name].append(result)
            return rows_by_method
        
        if self.config.max_workers > 1:
            pbar2 = tqdm(total=len(retrieval_results), desc=f"{desc} - 生成阶段")
            
            with ThreadPoolExecutor(max_workers=generation_workers) as generation_executor:
                futures = {generation_executor.submit(generation_phase, *r): r for r in retrieval_results}
                
                for future in as_completed(futures):
                    try:
                        method_name, result = future.result()
                        with rows_lock:
                            rows_by_method[method_name].append(result)
                    except Exception as exc:
                        logger.error("生成阶段异常: %s", exc)
                    
                    pbar2.update(1)
            
            pbar2.close()
        else:
            for method_name, sample, method_result, relevant_titles in tqdm(retrieval_results, desc=f"{desc} - 生成阶段"):
                method_name_g, result = generation_phase(method_name, sample, method_result, relevant_titles)
                with rows_lock:
                    rows_by_method[method_name].append(result)
        
        return rows_by_method

    def summarize_methods(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        return [
            {"Method": method_name, **self.aggregate_method_rows(rows)}
            for method_name, rows in rows_by_method.items()
        ]

    def aggregate_method_rows(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = [row["retrieval_metrics"] for row in rows]
        semantic = [row.get("semantic_metrics", {}) for row in rows]

        return {
            "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
            "Precision": round2(mean_or_none(m.precision for m in metrics)) or 0.0,
            "MRR": round2(mean_or_none(m.mrr for m in metrics)) or 0.0,
            "NDCG": round2(mean_or_none(m.ndcg for m in metrics)) or 0.0,
            "MAP": round2(mean_or_none(m.map_score for m in metrics)) or 0.0,
            "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
            "Time/ms": round2(mean_or_none(m.time_ms for m in metrics)) or 0.0,
            "Expanded Nodes": round2(mean_or_none(m.expanded_nodes for m in metrics)) or 0.0,
            "correctness": round2(mean_or_none(s.get("correctness") for s in semantic)),
            "faithfulness": round2(mean_or_none(s.get("faithfulness") for s in semantic)),
            "answer_relevance": round2(mean_or_none(s.get("answer_relevance") for s in semantic)),
            "context_relevance": round2(mean_or_none(s.get("context_relevance") for s in semantic)),
        }

    def semantic_record_table(self, rows_by_method: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        table = []
        for method, rows in rows_by_method.items():
            semantic = [row.get("semantic_metrics", {}) for row in rows]
            record = {"Method": method}
            for key in ["correctness", "faithfulness", "answer_relevance", "context_relevance"]:
                values = [s.get(key) for s in semantic]
                record[key] = round2(mean_or_none(values))
                record[f"{key}_ci95"] = round2(ci95(values))
            table.append(record)
        return table

    def complexity_stratified_table(self, proposed_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        buckets = {
            "low": [],
            "medium": [],
            "high": [],
        }
        for row in proposed_rows:
            score = float(row.get("complexity_score", 0.0))
            level = str(row.get("complexity_level") or complexity_level_for_score(score, self.config.complexity_threshold))
            buckets[level if level in buckets else complexity_level_for_score(score, self.config.complexity_threshold)].append(row)

        out = []
        for level, rows in buckets.items():
            metrics = [row["retrieval_metrics"] for row in rows]
            graph_count = sum(1 for row in rows if row.get("route") == "graph_expansion")
            out.append(
                {
                    "Complexity": level,
                    "Samples": len(rows),
                    "Graph Trigger Rate": round2(graph_count / len(rows) if rows else 0.0) or 0.0,
                    "Avg Len": round2(mean_or_none(m.avg_len for m in metrics)) or 0.0,
                    "Recall": round2(mean_or_none(m.recall for m in metrics)) or 0.0,
                }
            )
        return out

    def efficiency_table(self, method_summary: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        storage_map = {
            "Semantic RAG": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.documents_path),
            "+Rerank": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.documents_path),
            "GraphRAG": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.documents_path),
            "KG-RAG": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.documents_path),
            "MacRAG": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.paragraph_store_path) + dir_size_mb(self.config.documents_path),
            "Proposed": dir_size_mb(self.config.sentence_store_path) + dir_size_mb(self.config.paragraph_store_path) + dir_size_mb(self.config.documents_path),
        }
        return [
            {
                "Method": row["Method"],
                "Time/ms": round2(row.get("Time/ms", 0.0)) or 0.0,
                "Avg Len": round2(row.get("Avg Len", 0.0)) or 0.0,
                "Expanded Nodes": round2(row.get("Expanded Nodes", 0.0)) or 0.0,
                "Storage/MB": round2(storage_map.get(row["Method"], 0.0)) or 0.0,
            }
            for row in method_summary
        ]

    # ------------------------------------------------------------------
    # IO辅助函数
    # ------------------------------------------------------------------
    def _create_run_dir(self) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.config.output_dir / f"paper_hotpotqa_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    @staticmethod
    def get_relevant_titles(sample: pd.Series) -> Set[str]:
        supporting = sample.get("supporting_facts", {})
        if hasattr(supporting, "get"):
            titles = supporting.get("title", [])
        else:
            titles = []
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        return set(str(t) for t in titles)

    @staticmethod
    def get_relevant_facts(sample: pd.Series) -> Set[Tuple[str, str]]:
        supporting = sample.get("supporting_facts", {})
        if not hasattr(supporting, "get"):
            return set()
        titles = supporting.get("title", [])
        sent_ids = supporting.get("sent_id", [])
        if isinstance(titles, np.ndarray):
            titles = titles.tolist()
        if isinstance(sent_ids, np.ndarray):
            sent_ids = sent_ids.tolist()
        return {
            (str(title), str(sent_id))
            for title, sent_id in zip(titles, sent_ids)
            if str(title)
        }

    @staticmethod
    def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fieldnames: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    @classmethod
    def _json_safe(cls, obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, ExperimentMetrics):
            return asdict(obj)
        if isinstance(obj, EvidenceUnit):
            return asdict(obj)
        if isinstance(obj, dict):
            return {str(k): cls._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [cls._json_safe(v) for v in obj]
        return obj
