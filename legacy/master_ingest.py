import os
import json
import uuid
import csv
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- CONFIG ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "lexicore_debater_collection"
MODEL = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Wipe old data for a clean start
try:
    client.delete_collection(COLLECTION_NAME)
    print("🗑️ Deleted old collection for a clean re-ingest.")
except:
    pass

collection = client.get_or_create_collection(COLLECTION_NAME)

def ingest_batch(docs, metadatas, ids):
    if not docs:
        return
    
    # ChromaDB's hard limit is 5461. We use 5000 to be safe.
    SAFE_BATCH_SIZE = 5000
    total_items = len(docs)
    
    print(f"📦 Preparing to ingest {total_items} items...")

    for i in range(0, total_items, SAFE_BATCH_SIZE):
        # Calculate the end index for the current slice
        end = min(i + SAFE_BATCH_SIZE, total_items)
        
        batch_docs = docs[i:end]
        batch_metas = metadatas[i:end]
        batch_ids = ids[i:end]

        print(f"  -> Ingesting items {i} to {end}...")
        
        # Generate embeddings and add to collection
        collection.add(
            embeddings=MODEL.encode(batch_docs).tolist(),
            documents=batch_docs,
            metadatas=batch_metas,
            ids=batch_ids
        )
    
    print(f"✅ Finished ingesting {total_items} items.")

# --- 1. QURAN (CSV) ---
def process_quran():
    path = "./data/quran-english.csv"
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        docs, metas, ids = [], [], []
        for row in reader:
            docs.append(row['translation'])
            metas.append({"text_type": "Islamic", "scripture_source": f"Surah {row['surah_number']}:{row['verse_number']}"})
            ids.append(str(uuid.uuid4()))
        ingest_batch(docs, metas, ids)
        print(f"✅ Ingested Quran: {len(docs)} verses")

# --- 2. BIBLE (JSON) ---
def process_bible():
    path = "./data/lexicore_full_bible.json"
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs, metas, ids = [], [], []
        for entry in data:
            docs.append(entry['text_segment'])
            metas.append({"text_type": "Christian", "scripture_source": f"{entry['scripture_source']} {entry['citation_ref']}"})
            ids.append(entry.get('id', str(uuid.uuid4())))
        ingest_batch(docs, metas, ids)
        print(f"✅ Ingested Bible: {len(docs)} verses")

# --- 3. CREEDS (JSON) ---
def process_creeds():
    path = "./data/lexicore_creeds.json"
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        docs, metas, ids = [], [], []
        for entry in data:
            # Flexible check for 'text' or 'content' keys
            content = entry.get('text_segment') or entry.get('text') or entry.get('content')
            if content:
                docs.append(content)
                metas.append({"text_type": "Christian", "scripture_source": entry.get('scripture_source', 'Christian Creed')})
                ids.append(str(uuid.uuid4()))
        ingest_batch(docs, metas, ids)
        print(f"✅ Ingested Creeds: {len(docs)} segments")

# --- 4. SIRA (TXT) ---
def process_sira():
    path = "./data/sira.txt"
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_text(text)
    docs, metas, ids = [], [], []
    for chunk in chunks:
        docs.append(chunk)
        metas.append({"text_type": "Historical", "scripture_source": "Sira (Ibn Ishaq)"})
        ids.append(str(uuid.uuid4()))
    ingest_batch(docs, metas, ids)
    print(f"✅ Ingested Sira: {len(docs)} chunks")

# --- 5. HADITH (JSON) ---
def process_hadith():
    path = "./data/bukhari_sample.json"
    if not os.path.exists(path): return
    
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Fix: If the JSON was double-encoded as a string, load it again
    if isinstance(data, str):
        data = json.loads(data)
        
    docs, metas, ids = [], [], []
    
    # Fix: Ensure we are looping over a list of dictionaries
    for entry in data:
        if isinstance(entry, dict):
            # Safe access to 'text' or 'hadith_text'
            content = entry.get('text') or entry.get('hadith_text')
            if content:
                docs.append(content)
                metas.append({
                    "text_type": "Islamic", 
                    "scripture_source": f"Sahih Bukhari {entry.get('hadith_no', 'Unknown')}"
                })
                ids.append(str(uuid.uuid4()))
        else:
            # If entry is just a string (like a key), skip it or handle differently
            continue

    if docs:
        ingest_batch(docs, metas, ids)
        print(f"✅ Ingested Hadith: {len(docs)} entries")

# --- 6. POC DATA API (JSON) ---
def process_poc():
    path = "./data/lexicore_poc_data_api.json"
    if not os.path.exists(path):
        print(f"⚠️ POC file not found at {path}")
        return
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    docs, metas, ids = [], [], []
    
    for entry in data:
        # 1. Grab the text
        text = entry.get('text_segment', '')
        if not text: continue
        
        # 2. Map the metadata
        metas.append({
            "text_type": "Theological-POC",
            "scripture_source": entry.get('scripture_source', 'POC-API'),
            "citation_ref": entry.get('citation_ref', 'N/A'),
            "segment_type": entry.get('segment_type', 'Verse')
        })
        
        # 3. Use the existing ID or make a new one
        ids.append(entry.get('id', str(uuid.uuid4())))
        docs.append(text)

    # 4. We RE-EMBED here to ensure the vector "language" matches your other files
    if docs:
        print(f"🔄 Re-embedding and Ingesting POC Data ({len(docs)} segments)...")
        collection.add(
            embeddings=MODEL.encode(docs).tolist(),
            documents=docs,
            metadatas=metas,
            ids=ids
        )
        print("✅ POC Data Ingested.")

if __name__ == "__main__":
    print("🛡️ Starting Unified LexiCore Ingestion from ./data directory...")
    process_bible()
    process_quran()
    process_creeds()
    process_sira()
    process_hadith()
    process_poc()
    print(f"\n✨ SUCCESS: {collection.count()} segments integrated into the engine.")