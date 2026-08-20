# lexicore_search.py - LexiCore Semantic Search Engine (V1.0)

import json
import numpy as np
from sentence_transformers import SentenceTransformer
# We use numpy's dot product instead of scipy's cosine distance for cleaner code flow
# from scipy.spatial.distance import cosine 
import os

# --- 1. CONFIGURATION ---
DATA_FILE = 'lexicore_poc_data_api.json'
TOP_N_RESULTS = 5
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60' # Ensure model loading is robust

# --- 2. CORE FUNCTIONS ---

def load_data():
    """Loads the scripture segments and separates vectors and metadata."""
    print("--- Loading LexiCore Data ---")
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            segments = json.load(f)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data file '{DATA_FILE}' not found. Please run ingestion_pipeline.py first.")
        return None, None
    
    # Extract vectors and metadata into separate lists for efficient searching
    vectors = np.array([s['vector_embedding'] for s in segments])
    metadata = segments
    
    print(f"Data loaded successfully. Total segments available: {len(metadata)}")
    return vectors, metadata

def semantic_search(query: str, vectors, metadata, model, top_n: int):
    """
    Performs semantic search using cosine similarity (via dot product on normalized vectors).
    """
    print(f"\n--- Searching for: '{query}' ---")
    
    # 1. Embed the user query
    query_vector = model.encode([query])[0]
    
    # 2. Calculate Cosine Similarity 
    # Since Sentence-BERT vectors are already L2-normalized, the dot product 
    # is mathematically equal to the Cosine Similarity. This is fast and efficient.
    similarity_scores = np.dot(vectors, query_vector)
    
    # 3. Get the indices of the top results (highest similarity)
    # np.argsort returns indices that would sort the array; [::-1] reverses it for descending order.
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    
    # 4. Extract and format the results
    results = []
    for rank, idx in enumerate(top_indices):
        segment = metadata[idx]
        score = similarity_scores[idx]
        
        results.append({
            "Rank": rank + 1,
            "Source": segment['scripture_source'],
            "Citation": segment['citation_ref'],
            "Score": f"{score:.4f}",
            "Text": segment['text_segment'],
        })
        
    return results

def display_results(results):
    """Prints the formatted search results."""
    
    if not results:
        print("\nNo results found.")
        return
        
    print("\n" + "="*120)
    print(f"| {'Rank':<4} | {'Source':<15} | {'Citation':<25} | {'Score':<8} | {'Text':<60} |")
    print("="*120)
    
    for r in results:
        # Use a print structure that is easy to read in the console
        text_preview = r['Text'].replace('\n', ' ').strip()
        print(f"| {r['Rank']:<4} | {r['Source']:<15} | {r['Citation']:<25} | {r['Score']:<8} | {text_preview[:60]:<60} |")
    
    print("="*120 + "\n")
    
# --- 3. EXECUTION ---

if __name__ == "__main__":
    
    # *** Define the query here! This is the test for LexiCore's cross-faith search. ***
    USER_QUERY = "Why did God create man and the purpose of life"
    
    # Load Model
    try:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 
        print(f"Model loaded successfully: {EMBEDDING_MODEL._get_name()}")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Sentence Transformer model: {e}")
        exit()
        
    # Load Data
    vectors, metadata = load_data()
    if vectors is None:
        exit()
        
    # Run Search
    search_results = semantic_search(USER_QUERY, vectors, metadata, EMBEDDING_MODEL, TOP_N_RESULTS)
    
    # Display Results
    display_results(search_results)