from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Optional
import hashlib
import json
import re

import chromadb
from sentence_transformers import SentenceTransformer


SOURCE_ALIASES = {
    "bible": "Christian Scripture",
    "scripture": "Christian Scripture",
    "quran": "Islamic Scripture",
    "koran": "Islamic Scripture",
    "hadith": "Islamic Hadith",
    "bukhari": "Islamic Hadith",
    "muslim": "Islamic Hadith",
    "sira": "Islamic History",
    "church father": "Christian Patristic",
    "church fathers": "Christian Patristic",
    "creed": "Christian Creed",
    "historical": "Historical",
    "history": "Historical",
}

BIBLE_BOOKS = {
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges",
    "ruth", "1 samuel", "2 samuel", "1 kings", "2 kings", "1 chronicles", "2 chronicles",
    "ezra", "nehemiah", "tobit", "judith", "esther", "1 maccabees", "2 maccabees",
    "job", "psalms", "proverbs", "ecclesiastes", "song of songs", "wisdom", "sirach",
    "isaiah", "jeremiah", "lamentations", "baruch", "ezekiel", "daniel", "hosea", "joel",
    "amos", "obadiah", "jonah", "micah", "nahum", "habakkuk", "zephaniah", "haggai",
    "zechariah", "malachi", "matthew", "mark", "luke", "john", "acts", "romans",
    "1 corinthians", "2 corinthians", "galatians", "ephesians", "philippians", "colossians",
    "1 thessalonians", "2 thessalonians", "1 timothy", "2 timothy", "titus", "philemon",
    "hebrews", "james", "1 peter", "2 peter", "1 john", "2 john", "3 john", "jude", "revelation",
}


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def classify_source(source: str, metadata: Optional[dict[str, Any]] = None) -> tuple[str, str]:
    """Return (category, family). Explicit metadata always wins over heuristics."""
    metadata = metadata or {}
    explicit_category = _norm(metadata.get("category"))
    explicit_family = _norm(metadata.get("source_family"))
    if explicit_category:
        return explicit_category, explicit_family or "Other"

    text = _norm(source).lower()
    if any(k in text for k in ("quran", "koran", "surah")):
        return "Islamic Scripture", "Quran"
    if any(k in text for k in ("bukhari", "muslim", "tirmidhi", "abu dawud", "ibn majah", "hadith")):
        return "Islamic Hadith", "Hadith"
    if any(k in text for k in ("sira", "ibn ishaq", "ibn kathir", "tabari")):
        return "Islamic History", "Islamic History"
    if any(k in text for k in ("athanasian creed", "nicene creed", "apostles' creed", "apostles creed")):
        return "Christian Creed", "Creed"
    if any(k in text for k in ("clement", "ignatius", "polycarp", "irenaeus", "tertullian", "athanasius", "augustine", "chrysostom", "church father")):
        return "Christian Patristic", "Church Fathers"
    first = re.sub(r"[^a-z0-9 ]", "", text).split()
    if first and (first[0] in {b.split()[0] for b in BIBLE_BOOKS} or any(book in text for book in BIBLE_BOOKS)):
        return "Christian Scripture", "Bible"
    if "bible" in text or "scripture" in text:
        return "Christian Scripture", "Bible"
    if any(k in text for k in ("history", "historical", "chronicle", "annal")):
        return "Historical", "History"
    return "Other", "Other"


@dataclass
class Evidence:
    id: str
    text: str
    source: str
    category: str
    source_family: str
    citation: str
    distance: float
    relevance: float
    metadata: dict[str, Any]
    selected: bool = True

    def to_context(self, index: int) -> str:
        citation = self.citation or self.source
        return (
            f"EVIDENCE [{index}]\n"
            f"SOURCE: {self.source}\n"
            f"CATEGORY: {self.category}\n"
            f"CITATION: {citation}\n"
            f"TEXT: {self.text}\n"
            "END EVIDENCE"
        )


class EvidenceStore:
    def __init__(self, db_path: str, collection_name: str, embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.model = SentenceTransformer(embedding_model)
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(collection_name)

    def count(self) -> int:
        return self.collection.count()

    def metadata_schema(self, sample_size: int = 1000) -> dict[str, Any]:
        rows = self.collection.get(limit=sample_size, include=["metadatas"])
        fields: dict[str, dict[str, Any]] = {}
        for metadata in rows.get("metadatas") or []:
            if not metadata:
                continue
            for key, value in metadata.items():
                entry = fields.setdefault(key, {"types": set(), "examples": []})
                entry["types"].add(type(value).__name__)
                if len(entry["examples"]) < 5 and value not in entry["examples"]:
                    entry["examples"].append(value)
        return {
            key: {"types": sorted(value["types"]), "examples": value["examples"]}
            for key, value in fields.items()
        }

    def search(self, query: str, n_results: int = 12, category: Optional[str] = None,
               source_family: Optional[str] = None) -> list[Evidence]:
        query = _norm(query)
        if not query:
            return []
        embedding = self.model.encode(query, normalize_embeddings=True).tolist()
        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, min(n_results, 100)),
            "include": ["documents", "metadatas", "distances"],
        }
        filters = []
        if category:
            filters.append({"category": category})
        if source_family:
            filters.append({"source_family": source_family})
        if len(filters) == 1:
            kwargs["where"] = filters[0]
        elif len(filters) > 1:
            kwargs["where"] = {"$and": filters}

        result = self.collection.query(**kwargs)
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        output: list[Evidence] = []
        for i, doc in enumerate(docs):
            meta = metas[i] or {}
            source = _norm(meta.get("source") or meta.get("scripture_source") or meta.get("title") or "Unknown Source")
            category_value, family = classify_source(source, meta)
            category_value = _norm(meta.get("category")) or category_value
            family = _norm(meta.get("source_family")) or family
            distance = float(distances[i]) if i < len(distances) and distances[i] is not None else 1.0
            # Do not turn distance into fake probability. Relevance is only a normalized ranking indicator.
            relevance = 1.0 / (1.0 + max(distance, 0.0))
            citation = _norm(meta.get("citation") or meta.get("reference") or meta.get("scripture_reference"))
            output.append(Evidence(
                id=str(ids[i]), text=str(doc), source=source, category=category_value,
                source_family=family, citation=citation, distance=distance,
                relevance=relevance, metadata=meta,
            ))
        return output

    def search_balanced(self, query: str, n_results: int = 15, categories: Optional[list[str]] = None) -> list[Evidence]:
        """Retrieve without ideological boosts, then optionally guarantee category coverage."""
        categories = categories or []
        candidates = self.search(query, n_results=min(100, max(n_results * 3, 30)))
        if not categories:
            return candidates[:n_results]
        selected: list[Evidence] = []
        seen: set[str] = set()
        for category in categories:
            for item in candidates:
                if item.category == category and item.id not in seen:
                    selected.append(item)
                    seen.add(item.id)
                    break
        for item in candidates:
            if item.id not in seen:
                selected.append(item)
                seen.add(item.id)
            if len(selected) >= n_results:
                break
        return selected[:n_results]

    def build_context(self, evidence: Iterable[Evidence], max_chars: int = 30000) -> tuple[str, list[Evidence]]:
        chunks: list[str] = []
        selected: list[Evidence] = []
        total = 0
        for item in evidence:
            if not item.selected:
                continue
            chunk = item.to_context(len(selected) + 1)
            if total + len(chunk) > max_chars:
                break
            chunks.append(chunk)
            selected.append(item)
            total += len(chunk)
        return "\n\n".join(chunks), selected


def make_id(source: str, citation: str, text: str) -> str:
    raw = "|".join((_norm(source), _norm(citation), _norm(text)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    text = _norm(record.get("text") or record.get("document") or record.get("content"))
    source = _norm(record.get("source") or record.get("scripture_source") or record.get("title") or "Unknown Source")
    citation = _norm(record.get("citation") or record.get("reference") or record.get("scripture_reference"))
    category, family = classify_source(source, record)
    metadata = dict(record.get("metadata") or {})
    metadata.update({
        "source": source,
        "citation": citation,
        "category": category,
        "source_family": family,
        "ingestion_version": "2",
    })
    return {"id": record.get("id") or make_id(source, citation, text), "text": text, "metadata": metadata}


def load_json_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return payload["records"]
        if isinstance(payload.get("documents"), list):
            return payload["documents"]
    raise ValueError(f"Unsupported JSON structure: {path}")


def ingest_json(db_path: str, collection_name: str, json_path: str, embedding_model: str = "all-MiniLM-L6-v2") -> int:
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(name=collection_name, metadata={"schema_version": "2"})
    model = SentenceTransformer(embedding_model)
    normalized = [normalize_record(r) for r in load_json_records(Path(json_path))]
    normalized = [r for r in normalized if r["text"]]
    if not normalized:
        return 0
    embeddings = model.encode([r["text"] for r in normalized], normalize_embeddings=True).tolist()
    collection.upsert(
        ids=[r["id"] for r in normalized],
        documents=[r["text"] for r in normalized],
        metadatas=[r["metadata"] for r in normalized],
        embeddings=embeddings,
    )
    return len(normalized)
