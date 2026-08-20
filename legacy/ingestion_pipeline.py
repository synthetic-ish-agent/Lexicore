# ingestion_pipeline.py - LexiCore Data Ingestion Pipeline (V14 - FINAL)

from sentence_transformers import SentenceTransformer
import json
import os 
import requests 
from tqdm import tqdm 

# --- 1. CONFIGURATION ---
BIBLE_API_BASE = "https://bible-api.com/"
# FINAL FIX: Changing the API structure to use the generic path and specify translation in the function
QURAN_API_BASE = "http://api.alquran.cloud/v1/sura/" 
SEFARIA_API_BASE = "https://www.sefaria.org/api/texts/"

# Define the texts to process
BIBLE_BOOKS = {
    "GENESIS": 50, 
    "JOHN": 21     
}
SEFARIA_TEXTS = ["Genesis.1-10"] 
QURAN_SURAS = [1, 2, 3] 


# --- 2. INGESTION FUNCTIONS ---

## 2.1. BIBLE INGESTION (API)
def ingest_bible_api(books_with_chapters):
    """Ingests Bible data using a public API (bible-api.com)."""
    scripture_segments = []
    print("--- Starting Bible Parsing (via API) ---")
    
    for book_name, num_chapters in books_with_chapters.items(): 
        print(f"Processing {book_name} (via API)...")
        for chapter in tqdm(range(1, num_chapters + 1), desc=f"  {book_name}"):
            reference = f"{book_name} {chapter}"
            url = f"{BIBLE_API_BASE}{reference}?translation=kjv"
            
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status() 
                data = response.json()
                
                for verse_data in data.get("verses", []):
                    book_ref = f"{book_name} {verse_data['chapter']}:{verse_data['verse']}"
                    segment_unique_id = f"BIBLE_{book_name}_{verse_data['chapter']}_{verse_data['verse']}"

                    segment = {
                        "id": segment_unique_id,
                        "scripture_source": "Bible-POC", 
                        "original_language": "Unknown", 
                        "text_segment": verse_data['text'].strip(),
                        "citation_ref": book_ref,
                        "segment_type": "Verse",
                        "concept_tags": [], 
                    }
                    scripture_segments.append(segment)
                    
            except requests.exceptions.RequestException as e:
                print(f"Warning: Skipping {reference} due to API error: {e}")
                
    return scripture_segments

## 2.2. QURAN INGESTION (API)
def ingest_quran_api(suras_to_process):
    """Ingests Quran data (English translation) using alquran.cloud API."""
    scripture_segments = []
    print("\n--- Starting Quran Parsing (via API) ---")
    
    QURAN_TRANSLATION_ID = "en.yusufali" # Confirmed working translation ID

    for sura_number in tqdm(suras_to_process, desc="  Quran Suras"):
        # Corrected URL structure: /sura/{sura_number}/{translation_id}
        url = f"{QURAN_API_BASE}{sura_number}/{QURAN_TRANSLATION_ID}"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()['data']
            
            sura_name = data['englishName'] # Use English name for clarity
            
            for index, ayah in enumerate(data['ayahs']):
                ayah_number = index + 1
                ref = f"{sura_name} ({sura_number}):{ayah_number}"
                
                segment = {
                    "id": f"QURAN_{sura_number}_{ayah_number}",
                    "scripture_source": "Quran", 
                    "original_language": "Arabic", 
                    "text_segment": ayah['text'].strip(),
                    "citation_ref": ref,
                    "segment_type": "Ayah",
                    "concept_tags": [], 
                }
                scripture_segments.append(segment)
                
        except requests.exceptions.RequestException as e:
            print(f"Warning: Skipping Sura {sura_number} due to API error: {e}")
            
    return scripture_segments

## 2.3. SEFARIA INGESTION (API)
def ingest_sefaria_api(texts_to_process):
    """Ingests Torah/Talmud data using Sefaria's API."""
    scripture_segments = []
    print("\n--- Starting Sefaria Parsing (via API) ---")
    
    for text_ref in tqdm(texts_to_process, desc="  Sefaria Texts"):
        url = f"{SEFARIA_API_BASE}{text_ref}?context=0&commentary=0&pad=0"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            def flatten_text(data, segment_list, book_ref):
                if isinstance(data, str):
                    if data.strip():
                        segment_id = f"SEFARIA_{len(segment_list) + 1}"
                        
                        segment_list.append({
                            "id": segment_id,
                            "scripture_source": "Sefaria-Torah", 
                            "original_language": "Hebrew/Aramaic", 
                            "text_segment": data.strip(),
                            "citation_ref": book_ref,
                            "segment_type": "Segment",
                            "concept_tags": [], 
                        })
                elif isinstance(data, list):
                    for item in data:
                        flatten_text(item, segment_list, book_ref)

            flatten_text(data.get('text'), scripture_segments, data.get('ref', text_ref))
            
        except requests.exceptions.RequestException as e:
            print(f"Warning: Skipping Sefaria text {text_ref} due to API error: {e}")
            
    return scripture_segments

# --- 3. VECTORIZATION AND SAVE FUNCTION (Simplified and Combined) ---

def vectorize_and_save(segments, embedding_model):
    """Generates vector embeddings for segments and saves to a JSON file."""
    
    if not segments:
        print("No segments available for vectorization.")
        return

    batch_size = 128
    vectorized_segments = []
    
    print(f"\n--- Starting Vectorization for {len(segments)} Segments ---")
    
    for i in tqdm(range(0, len(segments), batch_size), desc="  Vectorizing"):
        batch = segments[i:i + batch_size]
        texts = [s["text_segment"] for s in batch]
        
        vectors = embedding_model.encode(texts).tolist()
        
        for j, segment in enumerate(batch):
            segment["vector_embedding"] = vectors[j]
            vectorized_segments.append(segment)
            
    print(f"\nVectorization complete. Total segments vectorized: {len(vectorized_segments)}")

    output_filename = 'lexicore_poc_data_api.json'
    with open(output_filename, 'w') as f:
         json.dump(vectorized_segments, f, indent=2)
    
    print(f"\nSUCCESS: LexiCore API data pipeline completed!")
    print(f"Output saved to: {output_filename}")


# --- 4. EXECUTION ---

if __name__ == "__main__":
    
    # 1. Initialize the AI Model
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
    try:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 
        print(f"Model loaded successfully from cache: {EMBEDDING_MODEL._get_name()}")
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Failed to load Sentence Transformer model. Check connectivity or installation: {e}")
        exit()
    
    all_segments = []
    
    # 2. Ingest Data from all Sources
    all_segments.extend(ingest_bible_api(BIBLE_BOOKS))
    all_segments.extend(ingest_quran_api(QURAN_SURAS))
    all_segments.extend(ingest_sefaria_api(SEFARIA_TEXTS))
    
    # 3. Vectorize and Save
    if all_segments:
        vectorize_and_save(all_segments, EMBEDDING_MODEL)
    else:
        print("\nPipeline finished, but no data segments were collected from any API.")