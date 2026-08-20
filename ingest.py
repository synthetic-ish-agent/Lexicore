from __future__ import annotations

import argparse, json, os
from pathlib import Path
from lexicore.loaders import load_all
from lexicore.store import EvidenceStore, DEFAULT_COLLECTION

p=argparse.ArgumentParser(description="Build the canonical LexiCore evidence index without touching any existing Chroma collection.")
p.add_argument("--data", default="./data")
p.add_argument("--db", default=os.getenv("LEXICORE_DB_PATH","./chroma_db"))
p.add_argument("--collection", default=DEFAULT_COLLECTION)
p.add_argument("--exclude-poc", action="store_true")
p.add_argument("--reset", action="store_true", help="Delete only the target canonical collection before rebuilding it.")
args=p.parse_args()

records=load_all(Path(args.data),include_poc=not args.exclude_poc)
store=EvidenceStore.open_or_create(args.db,args.collection)
if args.reset:
    try: store.delete_collection()
    except Exception: pass
    store=EvidenceStore.open_or_create(args.db,args.collection)
store.add_records(records)
summary={"records_loaded":len(records),"db":args.db,"collection":args.collection,"count":store.count()}
print(json.dumps(summary,indent=2))
