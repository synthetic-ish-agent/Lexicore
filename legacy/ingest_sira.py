import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Connect to your database
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("lexicore_debater_collection")

# 2. Load the Sira text
with open("./data/sira.txt", "r", encoding="utf-8") as f:
    raw_sira = f.read()

# Split the long text into 1000-character pieces with overlap
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_text(raw_sira)

# 4. Add to the Vector Database
print(f"🔄 Processing {len(chunks)} segments of the Sira...")

# We add in batches to avoid overloading the memory
batch_size = 100
for i in range(0, len(chunks), batch_size):
    batch_docs = chunks[i:i + batch_size]
    batch_ids = [f"sira_segment_{j}" for j in range(i, i + len(batch_docs))]
    batch_metas = [{"source": "Sira Ibn Ishaq", "type": "biography"}] * len(batch_docs)
    
    collection.add(
        documents=batch_docs,
        metadatas=batch_metas,
        ids=batch_ids
    )

print("✅ The Sira is now fully searchable in LexiCore!")