import json
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
import os
import time
import uuid # Needed for generating unique IDs

# --- Configuration ---

# Directory for the persistent ChromaDB storage
CHROMA_DB_PATH = "./chroma_db" 
# Name of the collection (the database) inside ChromaDB
CHROMA_COLLECTION_NAME = "lexicore_debater_collection"

# List of all JSON data files to be ingested
DATA_FILES = [
    "lexicore_full_bible.json",
    "lexicore_creeds.json",
    "lexicore_poc_data_api.json" 
]
# --- End Configuration ---

def create_and_ingest_data():
    """
    1. Loads the embedding model.
    2. Initializes (or recreates) the ChromaDB collection.
    3. Reads documents from all configured JSON files.
    4. Validates and converts data into LangChain Document format.
    5. Embeds and loads all documents into ChromaDB in batches.
    """
    documents = []
    
    print("--- 1. Initializing Embedding Model (all-MiniLM-L6-v2) ---")
    try:
        # Load the SentenceTransformer model for local embedding/query
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        print(f"FATAL: Error loading embedding model. Ensure connectivity or local cache. Error: {e}")
        return

    # 2. Connect to ChromaDB client and reset/get collection
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    
    # --- SAFETY CHECK: Delete old collection to ensure a clean re-ingestion ---
    try:
        # We delete the collection to ensure we start fresh with the latest data and mapping.
        client.delete_collection(name=CHROMA_COLLECTION_NAME)
        print(f"Successfully deleted existing collection: {CHROMA_COLLECTION_NAME}")
    except Exception:
        # Collection might not exist, which is fine.
        pass
    
    # Create the new collection using the same model's dimensions
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' is ready for data.")

    # 3. Load and Validate Data from all JSON files
    print("\n--- 3. Loading and Validating Data Files ---")
    
    for file_path in DATA_FILES:
        if not os.path.exists(file_path):
            print(f"Warning: Data file not found: {file_path}. Skipping.")
            continue
            
        print(f"Processing data from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error reading JSON file {file_path}: {e}")
            continue

        for item in loaded_data:
            # We are using your custom keys: 'scripture_source' and 'citation_ref'
            source = item.get("scripture_source")
            citation = item.get("citation_ref")
            text = item.get("text_segment")
            
            # Validation: Skip segments missing critical metadata
            if not source or not citation or not text:
                continue

            # Create the LangChain Document object
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "scripture_source": source,
                        "citation_ref": citation,
                        "segment_type": item.get("segment_type", "Unknown"),
                        # Ensure 'id' is used as the key for ChromaDB, if available
                        "id": item.get("id"),
                        "full_text": text
                    }
                )
            )

    if not documents:
        print("\nFATAL: No valid documents found to ingest. Exiting.")
        return

    print(f"\n--- 4. Data Processing Complete. Total valid segments to ingest: {len(documents)} ---")
    
    # 5. Embedding and Ingestion into ChromaDB (Batching for stability)
    print("--- 5. Generating Embeddings and Ingesting into ChromaDB (This may take a moment)...")
    
    # Prepare all data lists
    ids = [doc.metadata.get("id") or str(uuid.uuid4()) for doc in documents]
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]
    
    # Generate embeddings in a single batch (this is still efficient for SentenceTransformers)
    print("Generating all embeddings...")
    embeddings = embedding_model.encode(texts, convert_to_tensor=False).tolist()
    
    # Define a safe batch size (5000 is safe for the 5461 limit)
    BATCH_SIZE = 5000 
    
    # Calculate the number of batches needed
    total_segments = len(documents)
    num_batches = (total_segments + BATCH_SIZE - 1) // BATCH_SIZE
    
    # Process and ingest in batches
    print(f"\n--- 6. Ingesting into ChromaDB in {num_batches} Batches ---")

    for i in range(num_batches):
        start_index = i * BATCH_SIZE
        end_index = min((i + 1) * BATCH_SIZE, total_segments)
        
        # Slice the lists for the current batch
        batch_ids = ids[start_index:end_index]
        batch_texts = texts[start_index:end_index]
        batch_metadatas = metadatas[start_index:end_index]
        batch_embeddings = embeddings[start_index:end_index]
        
        print(f"Ingesting Batch {i + 1}/{num_batches}: Segments {start_index + 1} to {end_index}...")

        # Add the current batch to ChromaDB
        collection.add(
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    # Final count is performed after the loop finishes
    final_count = collection.count()
    print(f"\n--- Ingestion Complete. Total segments in DB: {final_count} ---")
    print("The RAG agent is now ready to use the expanded knowledge base.")

if __name__ == "__main__":
    start_time = time.time()
    create_and_ingest_data()
    end_time = time.time()
    print(f"Total ingestion time: {end_time - start_time:.2f} seconds.")