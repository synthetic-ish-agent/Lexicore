from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable


def norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def stable_id(source_family: str, citation: str, text: str) -> str:
    raw = "|".join((norm(source_family), norm(citation), norm(text)))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


BIBLE_BOOKS = {
    "genesis", "exodus", "leviticus", "numbers", "deuteronomy", "joshua", "judges", "ruth",
    "1 samuel", "2 samuel", "1 kings", "2 kings", "1 chronicles", "2 chronicles", "ezra", "nehemiah",
    "tobit", "judith", "esther", "1 maccabees", "2 maccabees", "job", "psalm", "psalms", "proverbs",
    "ecclesiastes", "song of songs", "wisdom", "sirach", "isaiah", "jeremiah", "lamentations", "baruch",
    "ezekiel", "daniel", "hosea", "joel", "amos", "obadiah", "jonah", "micah", "nahum", "habakkuk",
    "zephaniah", "haggai", "zechariah", "malachi", "matthew", "mark", "luke", "john", "acts", "romans",
    "1 corinthians", "2 corinthians", "galatians", "ephesians", "philippians", "colossians",
    "1 thessalonians", "2 thessalonians", "1 timothy", "2 timothy", "titus", "philemon", "hebrews", "james",
    "1 peter", "2 peter", "1 john", "2 john", "3 john", "jude", "revelation",
}


@dataclass
class Record:
    id: str
    text: str
    source: str
    source_family: str
    category: str
    citation: str
    segment_type: str = ""
    language: str = ""
    dataset: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def chroma_metadata(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_family": self.source_family,
            "category": self.category,
            "citation": self.citation,
            "segment_type": self.segment_type,
            "language": self.language,
            "dataset": self.dataset,
            "schema_version": "3",
        }

    def evidence_block(self, number: int) -> str:
        return (
            f"EVIDENCE {number} | ID={self.id}\n"
            f"SOURCE={self.source}\n"
            f"FAMILY={self.source_family}\n"
            f"CATEGORY={self.category}\n"
            f"CITATION={self.citation or 'Unspecified'}\n"
            f"TEXT={self.text}\n"
            "END EVIDENCE"
        )


def classify(source: str, dataset: str = "", segment_type: str = "") -> tuple[str, str]:
    s = norm(source).lower()
    d = norm(dataset).lower()
    if "quran" in s or "koran" in s or "quran" in d:
        return "Islamic Scripture", "Quran"
    if any(x in s for x in ("bukhari", "hadith", "tirmidhi", "abu dawud", "ibn majah", "muslim")) or "hadith" in d:
        return "Islamic Hadith", "Hadith"
    if "sira" in s or "sirat" in s or "ibn ishaq" in s or "islamic history" in d:
        return "Islamic History", "Sira / Islamic History"
    if "creed" in s or "creedal" in segment_type.lower():
        return "Christian Creed", "Creed"
    if any(x in s for x in ("clement", "ignatius", "polycarp", "irenaeus", "tertullian", "athanasius", "augustine", "chrysostom", "church father")):
        return "Christian Patristic", "Church Fathers"
    sl = re.sub(r"[^a-z0-9 ]", "", s)
    if "bible" in s or "scripture" in s or any(book in sl for book in BIBLE_BOOKS):
        return "Christian Scripture", "Bible"
    if "sefaria" in s or "torah" in s:
        return "Jewish Scripture / Commentary", "Sefaria / Torah"
    if any(x in s for x in ("history", "historical", "chronicle", "annal")):
        return "Historical / Other", "History"
    return "Historical / Other", "Other"


def make_record(*, id: str | None, text: str, source: str, citation: str, dataset: str,
                segment_type: str = "", language: str = "", extra: dict[str, Any] | None = None) -> Record | None:
    text = norm(text)
    if not text:
        return None
    source = norm(source) or "Unknown Source"
    citation = norm(citation)
    category, family = classify(source, dataset, segment_type)
    rid = norm(id) or stable_id(family, citation, text)
    return Record(
        id=rid, text=text, source=source, source_family=family, category=category,
        citation=citation, segment_type=norm(segment_type), language=norm(language),
        dataset=norm(dataset), metadata=extra or {},
    )


def dedupe(records: Iterable[Record]) -> list[Record]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Record] = []
    for r in records:
        key = (r.source_family, norm(r.citation).lower(), norm(r.text).lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out
