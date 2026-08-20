import os
import chromadb
# --- SURGICAL CHANGE 1: Switched OpenAI to Google Gemini ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangChain-native RAG imports
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List

# --- CONFIGURATION ---
CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "lexicore_debater_collection"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# FIX: Define LLM_MODEL globally to satisfy Pylance and the Chain logic
# Using the stable 'flash' name which works best with version="v1"
LLM_MODEL = "gemini-2.5-flash"

# --- 1. Custom Hybrid Retriever Class ---

class LexiCoreHybridRetriever(BaseRetriever):
    """
    Custom Retriever that implements the Hybrid RAG logic:
    1. Global Semantic Search (Top 5).
    2. Targeted Filtered Search (Top 1 from specific sources).
    """
    vectorstore: Chroma 
    targeted_sources: List[str] = ["Revelation", "Exodus", "Genesis", "John 1:1-18", "Athanasian Creed"]
    k: int = 10 

    def _get_relevant_documents(self, query: str) -> List[Document]:
        all_results = []
        seen_ids = set()

        # A. Global Search
        global_docs = self.vectorstore.similarity_search(query, k=5)
        for doc in global_docs:
            doc_id = doc.metadata.get('id', doc.page_content[:50])
            if doc_id not in seen_ids:
                seen_ids.add(doc_id)
                all_results.append(doc)
        
        # B. Targeted Search
        for source in self.targeted_sources:
            try:
                source_docs = self.vectorstore.similarity_search(
                    query, 
                    k=1, 
                    filter={"scripture_source": source}
                )
                if source_docs:
                    doc = source_docs[0]
                    doc_id = doc.metadata.get('id', doc.page_content[:50])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        all_results.append(doc)
            except Exception:
                pass
        
        return all_results[:self.k]

# --- 2. UTILITY: Formatting Documents ---

def format_docs(docs: List[Document]):
    """Turns Documents into a clean string for the Prompt."""
    formatted = []
    for doc in docs:
        source = doc.metadata.get('scripture_source', 'N/A')
        citation = doc.metadata.get('citation_ref', 'N/A')
        formatted.append(f"SOURCE: {source} ({citation})\nCONTENT: {doc.page_content}\n---")
    return "\n\n".join(formatted)

def get_retriever():
    print("--- Initializing LexiCore RAG Retriever (LangChain Native) ---")
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    vectorstore = Chroma(
        client=client,
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_function
    )
    return LexiCoreHybridRetriever(vectorstore=vectorstore)

# --- 3. LLM and DEBATE CHAIN SETUP ---

def create_debate_chain(retriever_obj):
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found. Use 'set GOOGLE_API_KEY=your_key' in CMD.")

    # Initialize Gemini with the stable 'v1' endpoint to avoid 404s
    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL, 
        temperature=0.7, 
        # version="v1"
    )

    DEBATE_PROMPT = """
    You are the LexiCore Debating Agent, a highly respected Theologian.
    Use the context below to defend and expose the truth of the query.
    
    CONTEXT:
    {context}

    USER QUERY:
    {query}
    """
    
    prompt = ChatPromptTemplate.from_template(DEBATE_PROMPT)
    
    # Final Chain Logic
    debate_chain = (
        {
            "context": retriever_obj | format_docs, 
            "query": RunnablePassthrough() 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return debate_chain

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        retriever_obj = get_retriever()
        chain = create_debate_chain(retriever_obj)
        
        print("\n=======================================================")
        print(" LexiCore Debating Agent Ready (Gemini 2.5 Flash Stable)")
        print("=======================================================\n")
        
        while True:
            user_query = input("Enter theological query (or 'quit'): \n> ")
            if user_query.lower() == 'quit': 
                break
            
            if not user_query.strip():
                continue

            print("\n--- Formulating Response... ---\n")
            response = chain.invoke(user_query)
            
            print("=========================================================")
            print(response)
            print("=========================================================\n")

    except Exception as e:
        print(f"\n[ERROR]: {e}")