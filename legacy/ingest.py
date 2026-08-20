import argparse
from rag import ingest_json

parser = argparse.ArgumentParser(description="Ingest normalized LexiCore evidence JSON into ChromaDB")
parser.add_argument("json_path")
parser.add_argument("--db", default="./chroma_db")
parser.add_argument("--collection", default="lexicore_debater_collection")
parser.add_argument("--model", default="all-MiniLM-L6-v2")
args = parser.parse_args()

count = ingest_json(args.db, args.collection, args.json_path, args.model)
print(f"Upserted {count} records into {args.collection}")
