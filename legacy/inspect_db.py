import argparse, json
from rag import EvidenceStore

p = argparse.ArgumentParser()
p.add_argument("--db", default="./chroma_db")
p.add_argument("--collection", default="lexicore_debater_collection")
args = p.parse_args()
store = EvidenceStore(args.db, args.collection)
print(json.dumps({"count": store.count(), "metadata_schema": store.metadata_schema()}, indent=2, ensure_ascii=False))
