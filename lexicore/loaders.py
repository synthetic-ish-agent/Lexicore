from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from .core import Record, make_record, dedupe, norm


def load_bible(path: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for x in data:
        r = make_record(id=x.get("id"), text=x.get("text_segment", ""), source=x.get("scripture_source", "Bible"),
                        citation=x.get("citation_ref", ""), dataset="lexicore_full_bible", segment_type=x.get("segment_type", "Verse"),
                        language=x.get("original_language", "English"), extra={"concept_tags": ", ".join(x.get("concept_tags") or [])})
        if r: out.append(r)
    return out


def load_quran(path: Path) -> list[Record]:
    out = []
    with path.open(encoding="utf-8-sig", newline="") as f:
        for x in csv.DictReader(f):
            citation = f"Surah {x.get('surah_number')}:{x.get('verse_number')} ({x.get('transliteration')})"
            r = make_record(id=f"QURAN_{x.get('surah_number')}_{x.get('verse_number')}", text=x.get("translation", ""),
                            source="Quran", citation=citation, dataset="quran-english", segment_type="Verse",
                            language="English translation", extra={"surah_number": int(x["surah_number"]), "verse_number": int(x["verse_number"]), "surah_name": x.get("transliteration", ""), "revelation_type": x.get("type", "")})
            if r: out.append(r)
    return out


def load_creeds(path: Path) -> list[Record]:
    out = []
    for x in json.loads(path.read_text(encoding="utf-8")):
        r = make_record(id=x.get("id"), text=x.get("text_segment") or x.get("text") or x.get("content", ""),
                        source=x.get("scripture_source", "Christian Creed"), citation=x.get("citation_ref", ""),
                        dataset="lexicore_creeds", segment_type=x.get("segment_type", "Creedal Statement"),
                        language=x.get("original_language", ""), extra={"concept_tags": ", ".join(x.get("concept_tags") or [])})
        if r: out.append(r)
    return out


def load_bukhari(path: Path) -> list[Record]:
    data = json.loads(path.read_text(encoding="utf-8"))
    hadiths = data.get("hadiths", data if isinstance(data, list) else [])
    out = []
    for x in hadiths:
        eng = x.get("english") or {}
        text = eng.get("text") or x.get("text") or x.get("hadith_text") or ""
        narrator = eng.get("narrator", "")
        if narrator and text and not text.startswith(narrator):
            text = f"{narrator} {text}"
        citation = f"Sahih al-Bukhari, hadith {x.get('id', 'unknown')}"
        r = make_record(id=f"BUKHARI_{x.get('id', 'unknown')}", text=text, source="Sahih al-Bukhari",
                        citation=citation, dataset="bukhari_sample", segment_type="Hadith", language="English translation",
                        extra={"book_id": x.get("bookId"), "chapter_id": x.get("chapterId"), "id_in_book": x.get("idInBook")})
        if r: out.append(r)
    return out


def load_sira(path: Path, chunk_size: int = 1200, overlap: int = 150) -> list[Record]:
    text = path.read_text(encoding="utf-8", errors="replace")
    # Paragraph-first chunking avoids slicing through every sentence while keeping chunks manageable.
    paragraphs = [norm(x) for x in re.split(r"\n\s*\n", text) if norm(x)]
    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 1 <= chunk_size:
            current = f"{current} {p}".strip()
        else:
            if current: chunks.append(current)
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {p}".strip()
    if current: chunks.append(current)
    out=[]
    for i, chunk in enumerate(chunks,1):
        r=make_record(id=f"SIRA_{i:05d}", text=chunk, source="Sira / Sirat Rasul Allah", citation=f"Sira chunk {i}",
                      dataset="sira", segment_type="Historical Text", language="English translation")
        if r: out.append(r)
    return out


def load_poc(path: Path) -> list[Record]:
    data=json.loads(path.read_text(encoding="utf-8"))
    out=[]
    for x in data:
        r=make_record(id=x.get("id"), text=x.get("text_segment", ""), source=x.get("scripture_source", "POC"),
                      citation=x.get("citation_ref", ""), dataset="lexicore_poc_data_api", segment_type=x.get("segment_type", ""),
                      language=x.get("original_language", ""), extra={"concept_tags": ", ".join(x.get("concept_tags") or [])})
        if r: out.append(r)
    return out


def load_all(data_dir: str | Path, include_poc: bool = True) -> list[Record]:
    d=Path(data_dir)
    records=[]
    for name, fn in [
        ("bible", lambda: load_bible(d/"lexicore_full_bible.json")),
        ("quran", lambda: load_quran(d/"quran-english.csv")),
        ("bukhari", lambda: load_bukhari(d/"bukhari_sample.json")),
        ("creeds", lambda: load_creeds(d/"lexicore_creeds.json")),
        ("sira", lambda: load_sira(d/"sira.txt")),
    ]:
        p = {"bible":"lexicore_full_bible.json","quran":"quran-english.csv","bukhari":"bukhari_sample.json","creeds":"lexicore_creeds.json","sira":"sira.txt"}[name]
        if (d/p).exists(): records.extend(fn())
    if include_poc and (d/"lexicore_poc_data_api.json").exists():
        records.extend(load_poc(d/"lexicore_poc_data_api.json"))
    return dedupe(records)
