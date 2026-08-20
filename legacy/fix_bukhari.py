import chromadb
from sentence_transformers import SentenceTransformer
import json
import os

def ingest_bukhari_manually(file_name):
    # 1. Setup Path
    db_path = "./chroma_db"
    actual_file_path = os.path.join("data", file_name) if os.path.exists(os.path.join("data", file_name)) else file_name

    if not os.path.exists(actual_file_path):
        print(f"❌ File not found at: {actual_file_path}")
        return

    # 2. Initialize Model and DB inside the script
    print("🤖 Loading Embedding Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    print(f"📂 Connecting to Database at {db_path}...")
    db_client = chromadb.PersistentClient(path=db_path)
    
    # Get the specific collection name we found earlier
    collection_name = "lexicore_debater_collection"
    collection = db_client.get_collection(name=collection_name)

    # 3. Read and Parse JSON
    print(f"📖 Reading JSON: {actual_file_path}")
    with open(actual_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert JSON entries to text strings
    if isinstance(data, list):
        # This assumes each item in your JSON list is a string or has a 'text' key
        chunks = [str(item.get('text', item)) if isinstance(item, dict) else str(item) for item in data]
    else:
        text = json.dumps(data)
        chunks = [text[i:i+700] for i in range(0, len(text), 700)]

    # 4. Process and Upload
    print(f"🔄 Generating embeddings for {len(chunks)} segments...")
    ids = [f"bukhari_json_{i}" for i in range(len(chunks))]
    metadatas = [{"scripture_source": "Sahih al-Bukhari", "category": "Hadith"} for _ in chunks]
    embeddings = model.encode(chunks).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=chunks
    )
    print(f"🎉 SUCCESS! Added {len(chunks)} Bukhari segments to {collection_name}!")

# --- RUN IT ---
if __name__ == "__main__":
    ingest_bukhari_manually("bukhari_sample.json")