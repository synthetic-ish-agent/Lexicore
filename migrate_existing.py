from __future__ import annotations
import argparse, os
from lexicore.core import make_record, dedupe
from lexicore.store import EvidenceStore
import chromadb

p=argparse.ArgumentParser(description="Safely copy an existing Chroma collection into the canonical schema. The source collection is never deleted.")
p.add_argument("--source-db",default=os.getenv("LEXICORE_DB_PATH","./chroma_db"))
p.add_argument("--source-collection",required=True)
p.add_argument("--target-db",default=os.getenv("LEXICORE_DB_PATH","./chroma_db"))
p.add_argument("--target-collection",default="lexicore_evidence_v3")
p.add_argument("--batch",type=int,default=500)
a=p.parse_args()

client=chromadb.PersistentClient(path=a.source_db)
src=client.get_collection(a.source_collection)
records=[]
offset=0
while True:
    page=src.get(limit=a.batch,offset=offset,include=["documents","metadatas"])
    docs=page.get("documents") or []; metas=page.get("metadatas") or []; ids=page.get("ids") or []
    if not docs: break
    for i,doc in enumerate(docs):
        m=metas[i] or {}
        r=make_record(id=ids[i],text=doc,source=m.get("source") or m.get("scripture_source") or m.get("title") or "Unknown",
                      citation=m.get("citation") or m.get("citation_ref") or m.get("reference") or "",dataset=m.get("dataset") or "migrated",
                      segment_type=m.get("segment_type", ""),language=m.get("language") or m.get("original_language") or "",extra=m)
        if r: records.append(r)
    offset += len(docs)
    if len(docs)<a.batch: break
records=dedupe(records)
store=EvidenceStore.open_or_create(a.target_db,a.target_collection)
store.add_records(records)
print(f"Migrated {len(records)} records into {a.target_collection}. Source collection was not modified.")
