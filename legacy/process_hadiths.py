import json
import chromadb

# 1. Setup Connection to your existing database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("lexicore_debater_collection")

def process_hadith_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    book_title = data['metadata']['english']['title']
    
    # We need three lists for ChromaDB: texts, metadatas, and unique IDs
    documents = []
    metadatas = []
    ids = []

    for h in data['hadiths']:
        content = f"Source: {book_title}\nNarrator: {h['english']['narrator']}\nText: {h['english']['text']}"
        
        documents.append(content)
        metadatas.append({
            "source": book_title,
            "hadith_id": h['id'],
            "type": "hadith"
        })
        # IDs must be unique strings
        ids.append(f"hadith_{h['bookId']}_{h['id']}")
    
    return documents, metadatas, ids

# 2. Run the process
docs, metas, ids = process_hadith_json('data/bukhari_sample.json')

# 3. Inject into ChromaDB
collection.add(
    documents=docs,
    metadatas=metas,
    ids=ids
)

print(f"✅ Successfully injected {len(docs)} Hadiths into the database.")