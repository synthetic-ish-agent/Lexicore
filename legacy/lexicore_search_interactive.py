# lexicore_search_interactive.py - Interactive LexiCore Semantic Search Engine

import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys 

# --- 1. CONFIGURATION ---
DATA_FILE = 'lexicore_poc_data_api.json'
TOP_N_RESULTS = 5
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

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
    
    vectors = np.array([s['vector_embedding'] for s in segments])
    metadata = segments
    
    print(f"Data loaded successfully. Total segments available: {len(metadata)}")
    return vectors, metadata

def semantic_search(query: str, vectors, metadata, model, top_n: int):
    """
    Performs semantic search using cosine similarity.
    """
    print(f"\n--- Searching for: '{query}' ---")
    
    # 1. Embed the user query
    query_vector = model.encode([query])[0]
    
    # 2. Calculate Cosine Similarity 
    similarity_scores = np.dot(vectors, query_vector)
    
    # 3. Get the indices of the top results (highest similarity)
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
        # Format for clean display
        text_preview = r['Text'].replace('\n', ' ').strip()
        print(f"| {r['Rank']:<4} | {r['Source']:<15} | {r['Citation']:<25} | {r['Score']:<8} | {text_preview[:60]:<60} |")
    
    print("="*120 + "\n")
    
# --- 3. EXECUTION ---

if __name__ == "__main__":
    
    # Load Model
    try:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 
        print(f"Model loaded successfully: {EMBEDDING_MODEL._get_name()}")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Sentence Transformer model: {e}")
        sys.exit(1)
        
    # Load Data
    vectors, metadata = load_data()
    if vectors is None:
        sys.exit(1)
        
    print("\n===============================================")
    print(" LexiCore Interactive Semantic Search Ready! ")
    print("===============================================")
    
    while True:
        try:
            user_query = input("Enter your query (or type 'quit' to exit): \n> ")
            
            if user_query.lower() == 'quit':
                break
                
            if not user_query.strip():
                print("Please enter a valid query.")
                continue

            # Run Search
            search_results = semantic_search(user_query, vectors, metadata, EMBEDDING_MODEL, TOP_N_RESULTS)
            
            # Display Results
            display_results(search_results)
            
        except Exception as e:
            print(f"An error occurred during search: {e}")
            break