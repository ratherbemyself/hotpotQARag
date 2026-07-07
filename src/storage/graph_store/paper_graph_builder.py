# -*- coding: utf-8 -*-
"""Build the paper-aligned structural and semantic Neo4j graph."""
from __future__ import annotations

import re
import hashlib
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Sequence


def normalize_key(value: Any) -> str:
    text = str(value or "").strip().strip('"').strip("'")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def make_id(prefix: str, value: str) -> str:
    normalized = normalize_key(value)
    slug = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_") or "unknown"
    plain_slug = re.sub(r"\s+", "_", normalized).strip("_")
    if plain_slug == slug:
        return f"{prefix}::{slug}"
    digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:10]
    return f"{prefix}::{slug}::{digest}"


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", str(text or "").strip())
    return [part.strip() for part in parts if part and part.strip()]


def _plain(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _as_list(value: Any) -> List[Any]:
    value = _plain(value)
    if value is None:
        return []
    if isinstance(value, (str, bytes)):
        return [value]
    try:
        return list(value)
    except TypeError:
        return [value]


def _sentences_to_text(value: Any) -> str:
    parts = []
    for sentence in _as_list(value):
        sentence = _plain(sentence)
        if isinstance(sentence, (list, tuple)):
            parts.extend(str(part).strip() for part in sentence if str(part).strip())
        elif str(sentence).strip():
            parts.append(str(sentence).strip())
    return " ".join(parts).strip()


def _remember_title(titles: "OrderedDict[str, Dict[str, str]]", title: Any, content: str = "") -> None:
    text = str(title or "").strip().strip('"')
    if not text:
        return
    key = normalize_key(text)
    if key not in titles:
        titles[key] = {"title": text, "content": content}
    elif content and not titles[key].get("content"):
        titles[key]["content"] = content


def collect_sample_titles(samples: Sequence[Dict[str, Any]]) -> "OrderedDict[str, Dict[str, str]]":
    """Collect titles used by the sampled HotpotQA rows for bounded real-flow tests."""
    titles: "OrderedDict[str, Dict[str, str]]" = OrderedDict()
    for sample in samples:
        supporting = sample.get("supporting_facts", {}) if hasattr(sample, "get") else {}
        if hasattr(supporting, "get"):
            for title in _as_list(supporting.get("title", [])):
                _remember_title(titles, title)

        context = sample.get("context", {}) if hasattr(sample, "get") else {}
        if hasattr(context, "get"):
            context_titles = _as_list(context.get("title", []))
            context_sentences = _as_list(context.get("sentences", []))
            for idx, title in enumerate(context_titles):
                sentence_group = context_sentences[idx] if idx < len(context_sentences) else []
                _remember_title(titles, title, _sentences_to_text(sentence_group))
    return titles


def select_paper_graph_records(
    records: Sequence[Dict[str, Any]],
    samples: Sequence[Dict[str, Any]] | None = None,
    include_titles: Sequence[str] | None = None,
) -> List[Dict[str, Any]]:
    """Select corpus records needed by a sampled real run while preserving sample order."""
    wanted = collect_sample_titles(samples or [])
    for title in include_titles or []:
        _remember_title(wanted, title)
    if not wanted:
        return list(records)

    by_title = {normalize_key(record.get("title")): record for record in records if record.get("title")}
    selected: List[Dict[str, Any]] = []
    seen = set()
    for key, item in wanted.items():
        if key in seen:
            continue
        if key in by_title:
            selected.append(by_title[key])
            seen.add(key)
        elif item.get("content"):
            selected.append({"title": item["title"], "sentence_total": item["content"], "triplets": []})
            seen.add(key)
    return selected


def prepare_paper_graph_rows(records: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    documents: List[Dict[str, Any]] = []
    sections: List[Dict[str, Any]] = []
    paragraphs: List[Dict[str, Any]] = []
    sentences: List[Dict[str, Any]] = []
    structure_edges: List[Dict[str, Any]] = []
    entities_by_key: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    semantic_links_by_key: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
    sentence_mentions_by_key: "OrderedDict[tuple[str, str], Dict[str, Any]]" = OrderedDict()
    related_edges_by_key: "OrderedDict[tuple[str, str, str], Dict[str, Any]]" = OrderedDict()

    for doc_index, item in enumerate(records):
        title = str(item.get("title") or f"untitled-{doc_index}").strip()
        content = str(item.get("sentence_total") or item.get("content") or "").strip()
        doc_id = make_id("doc", title)
        section_id = make_id("section", title)
        paragraph_id = f"paragraph::{normalize_key(title)}::0"

        triplets = item.get("triplets") or []
        core_entities: "OrderedDict[str, str]" = OrderedDict()
        relation_summary: List[str] = []
        for triplet in triplets:
            subject = str(triplet.get("Subject") or triplet.get("subject") or "").strip().strip('"')
            predicate = str(triplet.get("Predicate") or triplet.get("predicate") or "").strip()
            obj = str(triplet.get("Object") or triplet.get("object") or "").strip().strip('"')
            if not subject or not obj:
                continue
            source_key = normalize_key(subject)
            target_key = normalize_key(obj)
            if not source_key or not target_key or source_key == target_key:
                continue
            core_entities.setdefault(source_key, subject)
            core_entities.setdefault(target_key, obj)
            relation_summary.append(f"{subject} - {predicate} - {obj}" if predicate else f"{subject} - {obj}")

            entities_by_key.setdefault(source_key, {"key": source_key, "name": subject})
            entities_by_key.setdefault(target_key, {"key": target_key, "name": obj})
            semantic_links_by_key[(section_id, source_key)] = {"section_id": section_id, "entity_key": source_key}
            semantic_links_by_key[(section_id, target_key)] = {"section_id": section_id, "entity_key": target_key}
            related_edges_by_key[(source_key, predicate, target_key)] = {
                "source_key": source_key,
                "predicate": predicate,
                "target_key": target_key,
                "section_id": section_id,
            }

        documents.append({"id": doc_id, "title": title})
        sections.append(
            {
                "id": section_id,
                "document_id": doc_id,
                "title": title,
                "sentence_total": content,
                "core_entities": list(core_entities.values()),
                "relation_summary": relation_summary[:12],
            }
        )
        paragraphs.append(
            {
                "id": paragraph_id,
                "section_id": section_id,
                "title": title,
                "position": 0,
                "text": content,
            }
        )
        structure_edges.append({"type": "HAS_SECTION", "source_id": doc_id, "target_id": section_id})
        structure_edges.append({"type": "HAS_PARAGRAPH", "source_id": section_id, "target_id": paragraph_id})

        previous_sentence_id = ""
        for sent_index, sentence in enumerate(split_sentences(content)):
            sentence_id = f"sentence::{normalize_key(title)}::{sent_index}"
            sentences.append(
                {
                    "id": sentence_id,
                    "paragraph_id": paragraph_id,
                    "section_id": section_id,
                    "title": title,
                    "sent_id": sent_index,
                    "position": sent_index,
                    "text": sentence,
                }
            )
            structure_edges.append({"type": "HAS_SENTENCE", "source_id": paragraph_id, "target_id": sentence_id})
            if previous_sentence_id:
                structure_edges.append({"type": "NEXT_SENTENCE", "source_id": previous_sentence_id, "target_id": sentence_id})
            previous_sentence_id = sentence_id

            sentence_lower = sentence.lower()
            for entity_key in core_entities:
                if entity_key and entity_key in sentence_lower:
                    sentence_mentions_by_key[(sentence_id, entity_key)] = {
                        "sentence_id": sentence_id,
                        "entity_key": entity_key,
                    }

    return {
        "documents": documents,
        "sections": sections,
        "paragraphs": paragraphs,
        "sentences": sentences,
        "entities": list(entities_by_key.values()),
        "structure_edges": structure_edges,
        "semantic_links": list(semantic_links_by_key.values()),
        "sentence_mentions": list(sentence_mentions_by_key.values()),
        "related_edges": list(related_edges_by_key.values()),
    }


def _embedding_text(row: Dict[str, Any], content_field: str) -> str:
    title = str(row.get("title") or "").strip()
    content = str(row.get(content_field) or "").strip()
    if title and content:
        return f"{title}\n{content}"
    return title or content


def _embedding_vector_to_list(vector: Any) -> List[float]:
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return [float(value) for value in vector]


def attach_paper_graph_embeddings(
    rows: Dict[str, List[Dict[str, Any]]],
    embedding_client: Any,
    batch_size: int = 32,
) -> Dict[str, List[Dict[str, Any]]]:
    """Attach local embedding vectors to graph text nodes before Neo4j writes."""
    batch_size = max(1, int(batch_size or 32))
    targets: List[tuple[Dict[str, Any], str]] = []
    for collection_name, content_field in (
        ("sections", "sentence_total"),
        ("paragraphs", "text"),
        ("sentences", "text"),
    ):
        for row in rows.get(collection_name, []):
            text = _embedding_text(row, content_field)
            if text:
                targets.append((row, text))

    for start in range(0, len(targets), batch_size):
        batch = targets[start : start + batch_size]
        embeddings = embedding_client.embed([text for _, text in batch])
        for (row, _), vector in zip(batch, embeddings):
            row["embedding"] = _embedding_vector_to_list(vector)
    return rows


def chunked(items: Sequence[Dict[str, Any]], size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_paper_graph_schema(graph_store: Any) -> None:
    statements = [
        "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT paragraph_id IF NOT EXISTS FOR (n:Paragraph) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT sentence_id IF NOT EXISTS FOR (n:Sentence) REQUIRE n.id IS UNIQUE",
        "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (n:Entity) REQUIRE n.key IS UNIQUE",
        "CREATE INDEX section_title IF NOT EXISTS FOR (n:Section) ON (n.title)",
        "CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)",
    ]
    for statement in statements:
        graph_store.query(statement)


def _first_count(rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0
    value = rows[0].get("count", 0)
    return int(value or 0)


def clear_paper_graph(graph_store: Any, batch_size: int = 5000) -> None:
    """Clear Neo4j in bounded transactions before rebuilding the paper graph."""
    while True:
        deleted = _first_count(
            graph_store.query(
                """
                MATCH ()-[r]->()
                WITH r LIMIT $batch_size
                WITH collect(r) AS rels
                FOREACH (r IN rels | DELETE r)
                RETURN size(rels) AS count
                """,
                {"batch_size": batch_size},
            )
        )
        if deleted == 0:
            break

    while True:
        deleted = _first_count(
            graph_store.query(
                """
                MATCH (n)
                WITH n LIMIT $batch_size
                WITH collect(n) AS nodes
                FOREACH (n IN nodes | DELETE n)
                RETURN size(nodes) AS count
                """,
                {"batch_size": batch_size},
            )
        )
        if deleted == 0:
            break


def load_paper_graph_batch(graph_store: Any, rows: Dict[str, List[Dict[str, Any]]]) -> None:
    graph_store.query(
        """
        UNWIND $rows AS row
        MERGE (n:Document {id: row.id})
        SET n.title = row.title
        """,
        {"rows": rows["documents"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MERGE (n:Section {id: row.id})
        SET n.title = row.title,
            n.sentence_total = row.sentence_total,
            n.document_id = row.document_id,
            n.core_entities = row.core_entities,
            n.relation_summary = row.relation_summary
        FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
            SET n.embedding = row.embedding
        )
        """,
        {"rows": rows["sections"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MERGE (n:Paragraph {id: row.id})
        SET n.title = row.title,
            n.text = row.text,
            n.position = row.position,
            n.section_id = row.section_id
        FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
            SET n.embedding = row.embedding
        )
        """,
        {"rows": rows["paragraphs"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MERGE (n:Sentence {id: row.id})
        SET n.title = row.title,
            n.text = row.text,
            n.sent_id = row.sent_id,
            n.position = row.position,
            n.paragraph_id = row.paragraph_id,
            n.section_id = row.section_id
        FOREACH (_ IN CASE WHEN row.embedding IS NULL THEN [] ELSE [1] END |
            SET n.embedding = row.embedding
        )
        """,
        {"rows": rows["sentences"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MERGE (n:Entity {key: row.key})
        SET n.name = row.name
        """,
        {"rows": rows["entities"]},
    )

    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for edge in rows["structure_edges"]:
        by_type.setdefault(edge["type"], []).append(edge)
    label_pairs = {
        "HAS_SECTION": ("Document", "Section"),
        "HAS_PARAGRAPH": ("Section", "Paragraph"),
        "HAS_SENTENCE": ("Paragraph", "Sentence"),
        "NEXT_SENTENCE": ("Sentence", "Sentence"),
    }
    for rel_type, edge_rows in by_type.items():
        source_label, target_label = label_pairs[rel_type]
        graph_store.query(
            f"""
            UNWIND $rows AS row
            MATCH (source:{source_label} {{id: row.source_id}})
            MATCH (target:{target_label} {{id: row.target_id}})
            MERGE (source)-[:{rel_type}]->(target)
            """,
            {"rows": edge_rows},
        )

    graph_store.query(
        """
        UNWIND $rows AS row
        MATCH (s:Section {id: row.section_id})
        MATCH (e:Entity {key: row.entity_key})
        MERGE (s)-[:SEMANTIC_LINKS]->(e)
        """,
        {"rows": rows["semantic_links"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MATCH (s:Sentence {id: row.sentence_id})
        MATCH (e:Entity {key: row.entity_key})
        MERGE (s)-[:MENTIONS]->(e)
        """,
        {"rows": rows["sentence_mentions"]},
    )
    graph_store.query(
        """
        UNWIND $rows AS row
        MATCH (source:Entity {key: row.source_key})
        MATCH (target:Entity {key: row.target_key})
        MERGE (source)-[r:RELATED {predicate: row.predicate}]->(target)
        SET r.section_id = row.section_id
        """,
        {"rows": rows["related_edges"]},
    )


def load_paper_graph(
    graph_store: Any,
    records: Sequence[Dict[str, Any]],
    batch_size: int = 500,
    clear: bool = True,
    embedding_client: Any | None = None,
    embedding_batch_size: int = 1024,
) -> None:
    if clear:
        clear_paper_graph(graph_store, batch_size=batch_size)
    create_paper_graph_schema(graph_store)
    for batch in chunked(records, batch_size):
        rows = prepare_paper_graph_rows(batch)
        if embedding_client is not None:
            attach_paper_graph_embeddings(rows, embedding_client, batch_size=embedding_batch_size)
        load_paper_graph_batch(graph_store, rows)
