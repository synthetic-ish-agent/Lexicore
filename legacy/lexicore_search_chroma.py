import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import sys

# --- CONFIGURATION ---
CHROMA_COLLECTION_NAME = "lexicore_segments"
TOP_N_RESULTS = 5
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'

# --- CORE FUNCTIONS ---

def connect_to_chroma(collection_name: str):
    """Initializes Chroma client and returns the collection."""
    print("--- Connecting to ChromaDB ---")
    try:
        client = chromadb.PersistentClient(path="./chroma_data")
        collection = client.get_collection(name=collection_name)
        count = collection.count()
        print(f"Connection successful. Indexed segments: {count}")
        return collection
    except Exception as e:
        print(f"FATAL ERROR: Could not connect to ChromaDB or find collection. Error: {e}")
        print("Ensure ingestion_pipeline_chroma.py was run successfully.")
        sys.exit(1)


def semantic_search_chroma(query: str, collection, model, top_n: int):
    """
    Performs semantic search directly via the ChromaDB API.
    """
    print(f"\n--- Searching for: '{query}' ---")
    
    # 1. Embed the user query
    query_vector = model.encode([query]).tolist()
    
    # 2. Query the Vector Database
    results = collection.query(
        query_embeddings=query_vector,
        n_results=top_n
    )
    
    # Chroma returns a nested dictionary/list structure. We clean it up here.
    cleaned_results = []
    
    # The results are lists of lists because we passed a list of queries (one query)
    metadatas = results['metadatas'][0]
    documents = results['documents'][0]
    distances = results['distances'][0]
    
    for rank, (meta, doc, dist) in enumerate(zip(metadatas, documents, distances)):
        # Chroma returns distance; we convert it back to similarity (1 - distance) for presentation
        # Since we set the space to 'cosine', dist is cosine distance.
        similarity_score = 1 - dist 
        
        cleaned_results.append({
            "Rank": rank + 1,
            "Source": meta.get('scripture_source', 'N/A'),
            "Citation": meta.get('citation_ref', 'N/A'),
            "Score": f"{similarity_score:.4f}",
            "Text": doc,
        })
        
    return cleaned_results

# Reusing the display function from the previous script
def display_results(results):
    """Prints the formatted search results."""
    
    if not results:
        print("\nNo results found.")
        return
        
    print("\n" + "="*120)
    print(f"| {'Rank':<4} | {'Source':<15} | {'Citation':<25} | {'Score':<8} | {'Text':<60} |")
    print("="*120)
    
    for r in results:
        text_preview = r['Text'].replace('\n', ' ').strip()
        print(f"| {r['Rank']:<4} | {r['Source']:<15} | {r['Citation']:<25} | {r['Score']:<8} | {text_preview[:60]:<60} |")
    
    print("="*120 + "\n")


# --- EXECUTION ---

if __name__ == "__main__":
    
    # Load Model
    try:
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2") 
        print(f"Model loaded successfully: {EMBEDDING_MODEL._get_name()}")
    except Exception as e:
        print(f"FATAL ERROR: Failed to load Sentence Transformer model: {e}")
        sys.exit(1)
        
    # Connect to ChromaDB
    CHROMA_COLLECTION = connect_to_chroma(CHROMA_COLLECTION_NAME)
        
    print("\n===============================================")
    print(" LexiCore Interactive Semantic Search Ready! ")
    print(" (Powered by ChromaDB) ")
    print("===============================================")
    
    while True:
        try:
            user_query = input("Enter your query (or type 'quit' to exit): \n> ")
            
            if user_query.lower() == 'quit':
                break
                
            if not user_query.strip():
                print("Please enter a valid query.")
                continue

            # Run Search using Chroma
            search_results = semantic_search_chroma(user_query, CHROMA_COLLECTION, EMBEDDING_MODEL, TOP_N_RESULTS)
            
            # Display Results
            display_results(search_results)
            
        except Exception as e:
            print(f"An error occurred during search: {e}")
            break