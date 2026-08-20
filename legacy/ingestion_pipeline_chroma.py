import json
import chromadb
import uuid

# --- CONFIGURATION ---
DATA_FILE = 'lexicore_poc_data_api.json'
CHROMA_COLLECTION_NAME = "lexicore_segments"

def sanitize_metadata(metadata: dict) -> dict:
    """
    Cleans metadata values to ensure they are compatible with ChromaDB.
    Chroma only supports str, int, float, or bool values for metadata.
    This function converts any problematic types (like lists) into strings.
    """
    cleaned_metadata = {}
    for key, value in metadata.items():
        if isinstance(value, list):
            # Convert list to a comma-separated string
            cleaned_metadata[key] = ", ".join(map(str, value))
        elif isinstance(value, dict):
            # Convert dictionary to a string representation
            cleaned_metadata[key] = str(value)
        elif isinstance(value, (str, int, float, bool, type(None))):
            # Supported types
            cleaned_metadata[key] = value
        else:
            # Fallback for any other unexpected type
            cleaned_metadata[key] = str(value)
    return cleaned_metadata


def ingest_to_chroma(data_file: str, collection_name: str):
    """
    Loads data from the JSON file and pushes it into a ChromaDB collection.
    """
    print(f"--- Starting ChromaDB Ingestion: {collection_name} ---")
    
    # 1. Initialize Chroma Client (Will create a local 'chroma_data' folder)
    # Using PersistentClient so data is saved between sessions
    client = chromadb.PersistentClient(path="./chroma_data")
    
    # 2. Load Data from JSON
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file '{data_file}' not found. Run the original ingestion_pipeline.py first.")
        return

    # 3. Create/Get the Collection
    # Delete the old, empty collection first to ensure a clean start
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        # Ignore if the collection doesn't exist yet
        pass
        
    collection = client.get_or_create_collection(
        name=collection_name, 
        metadata={"hnsw:space": "cosine"} # Use cosine similarity for retrieval
    )

    # 4. Prepare Data for Batch Insertion
    ids = []
    embeddings = []
    documents = []
    metadatas = []
    
    for segment in segments:
        # Use a consistent ID or generate one if missing (using uuid for safety)
        segment_id = segment.get("id", str(uuid.uuid4()))
        
        # Extract metadata, excluding the fields we need separately
        raw_meta = {k: v for k, v in segment.items() if k not in ['vector_embedding', 'text_segment']}
        
        # APPLY THE SANITIZATION FUNCTION HERE
        cleaned_meta = sanitize_metadata(raw_meta)
        
        ids.append(segment_id)
        embeddings.append(segment['vector_embedding'])
        documents.append(segment['text_segment'])
        metadatas.append(cleaned_meta) # Use the cleaned metadata

    # 5. Insert Data in Batches (Chroma is often faster with batches)
    batch_size = 500
    total_segments = len(ids)
    
    for i in range(0, total_segments, batch_size):
        start_index = i
        end_index = min(i + batch_size, total_segments)
        batch_num = i // batch_size + 1
        
        print(f"  Inserting batch {batch_num} ({start_index+1}-{end_index} of {total_segments})...")
        
        try:
            collection.add(
                embeddings=embeddings[start_index:end_index],
                documents=documents[start_index:end_index],
                metadatas=metadatas[start_index:end_index],
                ids=ids[start_index:end_index]
            )
        except Exception as e:
            # If an error still occurs, print the specific error and the problem batch range
            print(f"FATAL ERROR during batch insert {batch_num}: {e}")
            
    
    count = collection.count()
    print(f"\nSUCCESS: Data ingestion complete!")
    print(f"Total documents indexed in ChromaDB collection '{collection_name}': {count}")
    print(f"Data stored in local folder: ./chroma_data")

if __name__ == "__main__":
    ingest_to_chroma(DATA_FILE, CHROMA_COLLECTION_NAME)