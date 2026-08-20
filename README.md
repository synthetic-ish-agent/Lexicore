# LexiCore v3 — canonical evidence architecture

This version consolidates the multiple ingestion/search implementations into one canonical evidence layer.

## Project layout

```text
LexiCore/
├── .streamlit/          # keep your existing Streamlit config/secrets
├── .venv/               # local virtual environment; do not redistribute
├── __pycache__/         # generated Python cache; safe to ignore
├── chroma_db/           # existing Chroma DB; preserve it
├── chroma_data/         # legacy/alternate Chroma DB; preserve it
├── data/                # supplied source corpus
├── lexicore/            # canonical application core
├── app.py
├── ingest.py
├── inspect_db.py
├── migrate_existing.py
└── requirements.txt
```

## Important safety rule

The canonical collection is `lexicore_evidence_v3`. The ingestion script never deletes an existing collection unless `--reset` is explicitly supplied for that same target collection. `migrate_existing.py` copies from an existing collection without deleting or modifying the source collection.

## Build a fresh canonical index

```powershell
python -m pip install -r requirements.txt
python ingest.py --data .\data --db .\chroma_db
```

If you want to exclude the POC dataset:

```powershell
python ingest.py --data .\data --db .\chroma_db --exclude-poc
```

## Inspect your existing databases

This does not load the embedding model and does not modify the database:

```powershell
python inspect_db.py --db .\chroma_db
python inspect_db.py --db .\chroma_data
```

To inspect one collection:

```powershell
python inspect_db.py --db .\chroma_db --collection YOUR_COLLECTION
```

## Migrate an existing collection safely

First inspect it and identify its collection name. Then:

```powershell
python migrate_existing.py --source-db .\chroma_db --source-collection YOUR_COLLECTION --target-db .\chroma_db
```

The source collection is not deleted.

## Run the application

Set the Gemini API key in your environment or `.streamlit/secrets.toml`, then:

```powershell
streamlit run app.py
```

## What changed

- One canonical schema for Bible, Quran, Hadith, Sira, creeds and auxiliary material.
- Stable IDs instead of random UUIDs for deterministic re-ingestion.
- Exact/deterministic deduplication.
- Explicit `category` and `source_family` metadata.
- Chroma metadata filtering.
- No ideological distance boosts.
- No fake confidence percentages derived from vector distance.
- Model citations are validated against the supplied evidence IDs.
- Structured Gemini output with Pydantic.
- Adversarial review is separate from the primary answer.
- Existing Chroma collections can be inspected/migrated without destructive operations.

## About voice

The old voice implementation is intentionally not part of the canonical evidence pipeline. It mixes Streamlit state, WebSocket transport, async Gemini Live code and browser audio handling. It should be integrated later as a separate UI/transport module so a voice failure cannot corrupt the research engine.
