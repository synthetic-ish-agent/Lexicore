import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("lexicore_debater_collection")

def comparative_search(query):
    print(f"\n🔎 SEARCHING ALL TEXTS FOR: '{query}'")
    
    # We pull 5 results to see how different sources compare
    results = collection.query(
        query_texts=[query],
        n_results=5
    )

    for i in range(len(results['documents'][0])):
        # Pull metadata safely
        meta = results['metadatas'][0][i]
        source = meta.get('source', 'Unknown Source')
        doc_type = meta.get('type', 'Text')
        content = results['documents'][0][i]
        
        print(f"\n🔰 [{source.upper()}] ({doc_type})")
        print(f"Content: {content[:400]}...") # Show first 400 chars
        print("-" * 30)

# TEST: Search for a concept mentioned in all your texts
comparative_search("The nature of Jesus (Isa) and his relationship to God")