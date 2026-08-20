# LexiCore project audit manifest

## Supplied corpus analyzed

- Bible JSON: 31,102 verse records
- Quran CSV: 6,236 verses across 114 surahs
- Sahih al-Bukhari sample: 188 hadith records
- Athanasian Creed dataset: 16 records
- Sira text: 1,259,731 bytes of supplied text
- POC/API dataset: 2,679 records (2,412 Bible-POC + 267 Sefaria-Torah)

## Important finding

The project previously contained multiple competing Chroma pipelines:

- `lexicore_debater_collection` in `chroma_db`
- `lexicore_segments` in `chroma_data`

They used different metadata conventions and ingestion logic. The new canonical target is:

`lexicore_evidence_v3`

Existing collections should be preserved until their contents are inspected and, if desired, migrated.
