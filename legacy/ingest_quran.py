import os
import chromadb
from sentence_transformers import SentenceTransformer
import uuid
import csv # Using the built-in CSV module

# --- CONFIGURATION ---
CHROMA_DB_PATH = "./chroma_db" 
CHROMA_COLLECTION_NAME = "lexicore_debater_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
QURAN_TEXT_FILE = "./data/quran-english.csv" 

# --- DATA PREPARATION (Customized for your CSV) ---

def load_and_parse_quran(file_path):
    """
    Reads the Quran CSV file using the confirmed column headers: 
    'surah_number', 'verse_number', and 'translation'.
    """
    print(f"Loading and parsing data from: {file_path}")
    documents = []
    
    # --- CONFIRMED COLUMN NAMES ---
    SURAH_COL = 'surah_number' 
    AYAH_COL = 'verse_number'
    TEXT_COL = 'translation' 
    # -----------------------------
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Use DictReader to read rows as dictionaries with header keys
            reader = csv.DictReader(f) 
            
            for row in reader:
                try:
                    # Extract data using the confirmed column names
                    surah_num = row[SURAH_COL].strip()
                    ayah_num = row[AYAH_COL].strip()
                    text = row[TEXT_COL].strip()

                    # Simple validation to skip empty rows/translations
                    if not surah_num or not ayah_num or not text:
                        continue
                        
                    # Create the document structure for ChromaDB
                    documents.append({
                        "text": text,
                        "metadata": {
                            "scripture_source": f"Surah {surah_num}",
                            "citation_ref": f"{surah_num}:{ayah_num}",
                            "text_type": "Islamic"
                        }
                    })
                except KeyError as e:
                    print(f"FATAL: Column name {e} not found in CSV. Please verify the column headers.")
                    return None
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")
                    
    except FileNotFoundError:
        print(f"FATAL: Source file not found at {file_path}. Did you save 'quran-english.csv' into the './data' folder?")
        return None
        
    print(f"Successfully parsed {len(documents):,} Ayahs.")
    return documents

# --- CHROMADB INGESTION (Modified for Batch Processing) ---

def ingest_data_to_chroma(documents):
    """Initializes the embedding model and inserts the new documents into ChromaDB 
       using batch processing to avoid exceeding the maximum batch size."""
    
    if not documents:
        print("No documents to ingest. Aborting.")
        return

    # Use a safe batch size, well below the limit of 5461
    BATCH_SIZE = 1000 
    
    print("Initializing embedding model...")
    try:
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME) 
    except Exception as e:
        print(f"Error loading embedding model: {e}")
        return

    print(f"Connecting to ChromaDB at {CHROMA_DB_PATH}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH) 
    
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception:
        print("Collection not found. Creating it now.")
        collection = client.create_collection(CHROMA_COLLECTION_NAME)

    initial_count = collection.count()
    total_new_segments = len(documents)
    print(f"Database currently holds {initial_count:,} segments (before ingestion).")
    print(f"Preparing to ingest {total_new_segments:,} new segments in batches of {BATCH_SIZE}...")
    
    # Loop through the documents in chunks of BATCH_SIZE
    for i in range(0, total_new_segments, BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        
        texts = [doc['text'] for doc in batch]
        metadatas = [doc['metadata'] for doc in batch]
        ids = [str(uuid.uuid4()) for _ in batch]
        
        print(f"Processing batch {i//BATCH_SIZE + 1} of {total_new_segments // BATCH_SIZE + 1} ({len(batch)} segments)...")
        
        try:
            # 1. Generate embeddings for the current batch
            embeddings = embedding_model.encode(texts, convert_to_tensor=False).tolist()
            
            # 2. Insert the current batch into ChromaDB
            collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            print(f"  > Batch {i//BATCH_SIZE + 1} inserted successfully.")
            
        except Exception as e:
            print(f"❌ ERROR DURING BATCH INSERTION (Batch {i//BATCH_SIZE + 1}): {e}")
            # You might want to log this error and continue, but for simplicity, we'll stop here.
            return

    final_count = collection.count()
    inserted_count = final_count - initial_count
    print(f"\n✅ SUCCESSFULLY INGESTED {inserted_count:,} NEW SEGMENTS.")
    print(f"Database now holds a total of {final_count:,} segments (Christian + Islamic).")


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure this part and load_and_parse_quran() remain the same
    # ... (rest of the script)
    os.makedirs("./data", exist_ok=True)
    
    print("--- Starting Quran Data Ingestion Process ---")
    
    # Assuming load_and_parse_quran is correct based on your column review
    quran_data = load_and_parse_quran(QURAN_TEXT_FILE)
    
    if quran_data:
        ingest_data_to_chroma(quran_data)
    
    print("--- Ingestion Process Complete ---")