import streamlit as st
import os
import av
import uuid
import time 
import json
import queue
import base64
import asyncio
import chromadb
import websocket
import threading
import numpy as np
from fpdf import FPDF
from google import genai
from google.genai import types
from fpdf.enums import XPos, YPos
from streamlit_autorefresh import st_autorefresh
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import ChatPromptTemplate
from streamlit_webrtc import webrtc_streamer, WebRtcMode
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit.runtime.scriptrunner import add_script_run_ctx
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

# --- CONFIGURATION & ENVIRONMENT SETUP ---
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60' 

# SECRETS HANDLING
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("❌ FATAL: GOOGLE_API_KEY not found in environment variables.")
    st.info("Please ensure you have a .env file or have set the variable in your terminal.")
    st.stop()  # This stops the app here so it doesn't crash with 404s later
CHROMA_DB_PATH = "./chroma_db" 
CHROMA_COLLECTION_NAME = "lexicore_debater_collection"

# --- SURGICAL CHANGE 3:
LLM_MODEL = "gemini-2.5-flash"

BACKGROUND_IMAGES = {
    "None (Default)": "",
    "Jesus - The Good Shepherd": "https://i.imgur.com/8Yv9D8I.jpeg", 
    "Divine Mercy": "https://upload.wikimedia.org/wikipedia/commons/e/e0/Divine_Mercy.jpg", 
    "Crucifixion": "https://images.unsplash.com/photo-1515600405234-90e879038d15?q=80&w=2048"
}

class VoiceDebaterProcessor:
    def __init__(self, session):
        self.session = session

    def recv_audio(self, frame: av.AudioFrame) -> av.AudioFrame:
        # 1. Convert to ndarray
        audio_ndarray = frame.to_ndarray()
        
        # 2. Resample from 48k to 16k (simple decimation)
        # If your mic is 48000Hz, taking every 3rd sample gives 16000Hz
        resampled_data = audio_ndarray[:, ::3].tobytes() 
        
        # 3. Use a safe way to run async in Streamlit's sync callback
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.session.send_realtime_input(
                    audio={"data": resampled_data, "mime_type": "audio/pcm;rate=16000"}
                ))
        except Exception:
            pass
            
        return frame

# --- CORE FUNCTIONS ---

def generate_pdf(transcript_text, sources=None):
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    class PDF(FPDF):
        def header(self):
            self.set_font('Helvetica', 'B', 15)
            self.set_text_color(20, 50, 100)
            self.cell(self.epw, 10, 'AMOR OF TRUTH', border=0, 
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
            self.ln(5)
            self.set_draw_color(37, 99, 235)
            # Draw line using margins to prevent overflow
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(10)

        def footer(self):
            self.set_y(-15)
            self.set_font('Helvetica', 'I', 8)
            self.set_text_color(100, 100, 100)
            self.cell(0, 10, f'Forensic Page {self.page_no()}', align='C')

    pdf = PDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # --- 1. CLEANING & SANITIZATION ---
    sanitized_text = transcript_text.replace('**', '').replace('*', '').replace('#', '')
    
    # Fix Unicode characters for Latin-1 compatibility
    clean_text = sanitized_text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u2018', "'").replace('\u201d', '"').replace('\u201c', '"')
    safe_text = clean_text.encode('latin-1', 'replace').decode('latin-1')
    
    # --- 2. MAIN BODY ---
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(w=pdf.epw, h=7, text=safe_text)

    # --- 3. VERIFIED SOURCE TRAIL (HARDENED) ---
    if sources:
        # Prevent source trail from starting at the very bottom of a page
        if pdf.get_y() > 240:
            pdf.add_page()
        else:
            pdf.ln(10)
            
        pdf.set_font("Helvetica", 'B', 11)
        pdf.set_text_color(20, 50, 100)
        pdf.cell(pdf.epw, 10, "OFFICIAL SOURCE TRAIL:", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        
        pdf.set_font("Helvetica", size=10)
        pdf.set_text_color(0, 0, 0)

        source_idx = 1
        for src in sources:
            if isinstance(src, dict) and src.get('selected', True):
                source_name = str(src.get('source', 'Unknown Document'))
                # Using epw ensures long source names wrap to next line instead of cutting off
                src_entry = f"[{source_idx}] {source_name}".encode('latin-1', 'replace').decode('latin-1')
                pdf.multi_cell(w=pdf.epw, h=6, text=src_entry)
                source_idx += 1

    return pdf.output()

# --- CSS INJECTION FUNCTION (Includes Background Feature) ---
def apply_ultra_style():
    """Injects high-end CSS for a theological war-room aesthetic + Particles."""
    st.markdown("""
        <style>
        /* This targets the 'Running...' toast and the spinner specifically */
        [data-testid="stStatusWidget"], [data-testid="stToast"] {
            visibility: hidden !important;
            display: none !important;
        }
        /* 1. HIDE DEFAULT ELEMENTS */
        div[data-testid="stStatusWidget"] { 
            visibility: hidden;
            height: 0;
            position: fixed;
            display: none !important;
        }
        .stDeployButton { display:none; }

        /* 2. PARTICLE BACKGROUND CANVAS */
        particles-js {
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            z-index: -1; /* Stay behind everything */
            background: radial-gradient(circle, #0f172a 0%, #020617 100%);
        }

        /* 3. STARTUP CENTERING */
        .startup-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            height: 70vh;
            text-align: center;
        }

        /* 4. BUTTONS & CARDS */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white; border: none; padding: 0.75rem 2rem;
            border-radius: 12px; font-weight: 500;
            box-shadow: 0 0 15px rgba(124, 58, 237, 0.4);
            transition: all 0.3s ease;
        }
        
        .stApp { background: transparent; } /* Let particles show through */
        </style>

        <div id="particles-js"></div>
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
            particlesJS("particles-js", {
                "particles": {
                    "number": {"value": 100, "density": {"enable": true, "value_area": 800}},
                    "color": {"value": "#ffffff"},
                    "shape": {"type": "circle"},
                    "opacity": {"value": 0.5, "random": true},
                    "size": {"value": 3, "random": true},
                    "line_linked": {"enable": true, "distance": 150, "color": "#3b82f6", "opacity": 0.2, "width": 1},
                    "move": {"enable": true, "speed": 1.5, "direction": "none", "out_mode": "out"}
                }
            });
        </script>
    """, unsafe_allow_html=True)

# --- 1. RAG RETRIEVER SETUP (Upgraded with Source Filtering) ---
def classify_source(source_name):
    """Classifies a source as 'Christian', 'Islamic', or 'Historical/Other'."""
    source_name = source_name.strip()
    if source_name.startswith(("Surah", "Bukhari", "Quran")):
        return "Islamic"

    biblical_books = [
        "Revelation", "Exodus", "Genesis", "John", "Matthew", "Mark", 
        "Luke", "Acts", "Romans", "Corinthians", "Galatians", "Ephesians", 
        "Philippians", "Colossians", "Thessalonians", "Timothy", "Titus", 
        "Philemon", "Hebrews", "James", "Peter", "Jude"
    ]
    creedal_texts = ["Athanasian Creed", "Nicene Creed", "Apostles Creed"]

    main_source = source_name.split()[0].replace(':', '').replace('.', '') 
    if main_source in biblical_books or main_source in creedal_texts:
        return "Christian"
    return "Historical/Other"

@st.cache_resource
def get_retriever():
    try:
        # Safety Log 1
        print("DEBUG: Starting SentenceTransformer initialization...")
        
        # Safety Log 2
        print(f"DEBUG: Connecting to ChromaDB at {CHROMA_DB_PATH}...")
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH) 
        
        # Safety Log 3
        try:
            collection = client.get_collection(CHROMA_COLLECTION_NAME)
        except Exception as coll_err:
            st.error(f"❌ Database Error: Could not find collection '{CHROMA_COLLECTION_NAME}'. Did you run the ingestion script?")
            st.stop()

    except Exception as e:
        st.error(f"⚠️ Retriever Init Critical Error: {e}") 
        return None
    
    # Create a queue to hold audio chunks coming back from Gemini
if "audio_out_queue" not in st.session_state:
    st.session_state.audio_out_queue = queue.Queue()

def gemini_listener(session):
    """Background task to listen for Gemini's voice responses."""
    import asyncio
    async def listen():
        async for message in session:
            # Check if the message contains audio data
            if message.server_content and message.server_content.model_turn:
                parts = message.server_content.model_turn.parts
                for part in parts:
                    if hasattr(part, 'inline_data') and part.inline_data:
                        # Put the raw audio bytes into our queue
                        audio_bytes = part.inline_data.data
                        st.session_state.audio_out_queue.put(base64.b64encode(audio_bytes).decode())
    
    # Run the async listener in the background
    asyncio.run(listen())

# --- 1. RAG RETRIEVER SETUP ---
@st.cache_resource(show_spinner=False)
def load_resources():
    """Initializes the model and database connection once, silently."""
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        db_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        db_collection = db_client.get_collection(CHROMA_COLLECTION_NAME)
        return model, db_collection
    except Exception as e:
        st.error(f"❌ Database Initialization Failed: {e}")
        return None, None

async def start_voice_debate():
    # Use the genai client we imported
    client = genai.Client(api_key=GOOGLE_API_KEY, http_options={'api_version': 'v1alpha'})
    
    # Ensure LLM_MODEL is set to "gemini-2.0-flash-exp" or similar for Live API
    async with client.aio.live.connect(model=LLM_MODEL, config=get_live_config()) as session:
        st.write("🎙️ Live Connection Established. Start speaking...")
        # (WebSocket logic goes here)

def retrieve_segments(query_text):
    if 'embedding_model' not in st.session_state:
        model, coll = load_resources()
        st.session_state.embedding_model = model
        st.session_state.collection = coll
        
    embedding_model = st.session_state.embedding_model
    collection = st.session_state.collection

    if collection is None:
        return []
        
    try:
        # 1. Encode query
        query_embedding = embedding_model.encode(query_text).tolist()
        all_results = []
        
        # 2. Wide Net Search
        wide_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=25, 
            include=['documents', 'metadatas', 'distances']
        )
        
        if wide_results and wide_results.get('documents'):
            for i in range(len(wide_results['documents'][0])):
                all_results.append({
                    "doc": wide_results['documents'][0][i],
                    "meta": wide_results['metadatas'][0][i],
                    "id": wide_results['ids'][0][i],
                    "distance": wide_results['distances'][0][i]
                })

        # 3. Surgical Strike Search
        targeted_sources = [
            "John", "Matthew", "Isaiah", "Psalms", # Christian
            "Surah", "Quran", "Bukhari", "Hadith", # Islamic
            "Sira", "Ibn Ishaq", "Creed", "Athanasian" # History/Theology
        ]
        
        for source in targeted_sources:
            try:
                source_res = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=5,
                    where={"scripture_source": source}
                )
                if source_res and source_res['documents'] and len(source_res['documents'][0]) > 0:
                    for i in range(len(source_res['documents'][0])):
                        all_results.append({
                            "doc": source_res['documents'][0][i],
                            "meta": source_res['metadatas'][0][i],
                            "id": source_res['ids'][0][i],
                            "distance": source_res['distances'][0][i]
                        })
            except Exception:
                continue

        # 4. Deduplicate and Apply "Trilogos" Weighting
        unique_results = {}
        for item in all_results:
            dist = item['distance']
            src = str(item['meta'].get('scripture_source', '')).lower()
            
            # --- PRIORITY BOOST ---
            if any(x in src for x in ["surah", "quran", "bukhari", "hadith", "sira"]):
                dist -= 0.1
            
            if item['id'] not in unique_results or dist < unique_results[item['id']]['distance']:
                item['distance'] = dist
                unique_results[item['id']] = item

        return sorted(unique_results.values(), key=lambda x: x['distance'])
        
    except Exception as e:
        print(f"DEBUG Search Error: {e}")
        return []

def load_preset_query(query_text):
    st.session_state['query_input_text'] = query_text
    st.session_state.run_generation = True
    st.session_state.debate_result = None

# --- 2. LLM Chains ---
def create_adversarial_query_expander(llm_temperature):
    AQE_PROMPT = """
    You are the Devil's Advocate. Your task is to take a core theological concept or question and generate 3 to 5 common, critical, or skeptical counter-arguments/questions that an unbeliever would raise.
    Instructions:
    - **Do not** answer the query.
    - **Output a single, concise list** of these skeptical counter-queries, separated by semicolons.
    User Query: {query}
    Skeptical Counter-Queries (semicolon-separated list):
    """
    prompt = ChatPromptTemplate.from_template(AQE_PROMPT)
    # --- SURGICAL CHANGE 4: ---
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=llm_temperature)
    return prompt | llm | StrOutputParser()

def create_weakness_assessor(llm_temperature):
    CWA_PROMPT = """
    You are a Ruthless Adversarial Debater. 
    Analyze the argument and return ONLY a raw JSON object.

    Keys: 
    "weakest_points": (string)
    "defense_strategy": (string)
    
    Argument: {argument}
    Context: {context}
    """
    prompt = ChatPromptTemplate.from_template(CWA_PROMPT)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=llm_temperature)
    
def get_debate_prompt(debate_stance, citation_mode):
    """Returns the system prompt based on the selected stance and citation style."""
    
    # Handle Citation Style
    if citation_mode == "Minimalist":
        citation_instruction = "Use only the book and chapter/verse (e.g., Jn 1:1)."
    elif citation_mode == "Detailed Footnote Style":
        citation_instruction = "Provide detailed citations as footnotes at the end of your response."
    else:  # Inline (Scholarly)
        citation_instruction = "Ensure citations are concise, combining Source and Citation into parentheses (e.g., Job 1:22)."

        # NEW: UNIVERSAL FORMATTING & SANITIZATION RULES
    # This prevents the asterisks (*) from appearing in the response and PDF
    formatting_rules = """
    # CRITICAL FORMATTING RULES:
    1. NO MARKDOWN: Do not use asterisks (**) for bold or (*) for italics.
    2. NO HASHES: Do not use '#' for headers. Use CAPITALIZED PLAIN TEXT for sections.
    3. SANITIZED TEXT: Ensure the output is clean plain text compatible with standard PDF encoding (Latin-1).
    """

    # 1. DIDACTIC MODE: For teaching core Christian truth.
    if debate_stance == "Didactic/Explanatory":
        return f"""
        You are the LexiCore Didactic Agent, a gentle teacher of Christian(catholic) doctrine.
        Your goal is to explain theology clearly and correct misconceptions like 'Partialism' (God is not 1/3 segments).
        
        {citation_instruction}
        
        # STRUCTURE:
        1. Definition: Define the concept (e.g., the Trinity is One Essence in Three Persons).
        2. - If the user asks about 'Orthodox', 'Church', 'Christian', 'Christ', 'Cross', or 'Trinity', ONLY use sources labeled as 'Bible' or 'Church Fathers'.
            - Do NOT use Islamic Surahs to support Christian doctrine unless the user explicitly asks for a comparison.
            - Use the 'scripture_source' metadata to filter your response.
        3. Correction: Specifically debunk any heresy or misunderstanding in the query.
        4. Christ-Centered Focus: Explain how this truth leads to the peace and love of Jesus.
        5. Let us prayer: Pray for the person for more wisdom and understanding, and God's guildance towards the question.
        
        # CONTEXT:
        {{context}}
        # QUERY:
        {{query}}
        """

    # 2. SCHOLARLY DEBATE: Academic defense and Internal Critique of Islam.
    elif debate_stance == "Scholarly (Debate)":
        return f"""
        You are the LexiCore Scholarly Debater. Your tone is forensic, firm, and academic.
        You defend the Trinity and the Deity of Christ using Systematic Theology.
        
        {citation_instruction}
        
        # DEBATE STRATEGY:
        - DIRECT RESPONSE: Begin with a 1-sentence direct affirmation or denial of the premise.
        - THE ISLAMIC DILEMMA: If the query is Islamic, use the Quran from the CONTEXT to prove the Bible. (e.g., Surah 5:47, 10:94), if not, use only the Bible from the CONTEXT to prove your point.
        - If it is about Gospel and If the Quran confirms the Gospel, and the Gospel says Jesus is God, then the Quran contradicts itself by denying His deity.
        - CREED EVIDENCE (MANDATORY): Do NOT just mention the name (e.g., Athanasian Creed). You MUST extract and quote the actual theological clauses from the creeds provided in the CONTEXT, if the debate is about trinity.
        - LOGICAL DEFENSE: Address the 'Teaching' paradox by explaining the Economic vs Ontological Trinity.
        - EVIDENCE: Rely on early creeds (Athanasian/Nicene) to define terms.
        
        # CONTEXT:
        {{context}}
        # CHALLENGE:
        {{challenge}}
        # COUNTER-STRATEGY:
        {{counter_strategy}}
        # QUERY:
        {{query}}
        """

    # 3. SKEPTICAL/CONTRARIAN: Offensive Apologetics & Critique of Islam.
    else:
        return f"""
        You are a Forensic Apologist. Your goal is to prove Christianity is the only truth by exposing the human origins and moral failures of Islam.
        
        {citation_instruction}
        
        # OFFENSIVE APOLOGETICS MANDATE:
        - CRITIQUE MUHAMMAD: Contrast the sinlessness and sacrificial love of Jesus with Muhammad's military conquests, raids, and moral controversies (e.g., marriage to Aisha or Zaynab).
        - EXPOSE CONTRADICTIONS: Point out abrogation (Naskh) in the Quran where Allah changes his mind. Mention historical errors (Haman in Egypt, Mary as Moses' sister).
        - MORAL CONTRAST: Contrast the "Sword Verses" (Surah 9:5) with Jesus' command to love enemies. Prove that Islam is a religion of submission to a man, while Christianity is a relationship with the God of Love.
        - THE TRINITY: Forcefully reject the '1/3' claim. God is Spirit; He is not divisible.
        
        # CONTEXT:
        {{context}}
        # CHALLENGE:
        {{challenge}}
        # COUNTER-STRATEGY:
        {{counter_strategy}}
        # QUERY:
        {{query}}
        """
def create_debate_chain(_retriever_func, llm_temperature, debate_stance, citation_mode): 
    if not os.getenv("GOOGLE_API_KEY"):
        return None
    # --- SURGICAL CHANGE 6: Switched to ChatGoogleGenerativeAI ---
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0.7,
        # 'thinking_budget' allows the model to process 
        thinking_budget=1024 
    )
    DEBATE_PROMPT = get_debate_prompt(debate_stance, citation_mode)
    prompt = ChatPromptTemplate.from_template(DEBATE_PROMPT)
    
    debate_chain = (
        {
            "context": RunnablePassthrough(), 
            "query": RunnablePassthrough(),
            "challenge": RunnablePassthrough(),
            "counter_strategy": RunnablePassthrough() 
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return debate_chain

# --- 3. UI UTILITY FUNCTIONS ---
def get_context_for_llm():
    """Compiles the user-selected context segments into a single string for the LLM."""
    if 'retrieved_candidates' not in st.session_state:
        return ""
    
    context_strings_for_llm = []
    for item in st.session_state['retrieved_candidates']:
        if item['selected']:
            context_string = (
                f"CONTEXT SEGMENT (Source: {item['source']} - Distance: {item['distance']:.4f}):\n" 
                f"'{item['text']}'\n---\n"
            )
            context_strings_for_llm.append(context_string)
            
    return "\n".join(context_strings_for_llm).strip()

def calculate_context_utilization(response, context_list):
    """Calculates a simple percentage of how many unique selected contexts were used."""
    if not context_list or not response:
        return 0, 0
        
    used_count = 0
    selected_count = sum(1 for item in context_list if item['selected'])
    
    if selected_count == 0:
        return 0, 0
    
    for item in context_list:
        if item['selected']:
            source_ref = item['source'] 
            # Logic to extract book and cite for verification
            parts = source_ref.split()
            source_book = parts[0].replace(':', '').replace('.', '') if parts else ""
            source_cite = parts[-1].replace(':', '').replace('.', '') if len(parts) > 1 else ""
            
            if source_book in response and source_cite in response:
                used_count += 1
                
    utilization_score = (used_count / selected_count) * 100 if selected_count > 0 else 0
    return utilization_score, used_count


# --- 4. PERSISTED DISPLAY FUNCTIONS ---
def display_agent_response(response_placeholder):
    """Displays the response and the forensic trail of sources."""
    response = st.session_state.get('last_response')
    candidates = st.session_state.get('retrieved_candidates', [])

    # Define color and icon mapping for different sources
    source_map = {
        "Sahih al-Bukhari": {"color": "#2ECC71", "icon": "📜", "label": "Hadith"}, 
        "Surah": {"color": "#27AE60", "icon": "📖", "label": "Quran"},           
        "John": {"color": "#3498DB", "icon": "✝️", "label": "Bible"},            
        "Matthew": {"color": "#3498DB", "icon": "✝️", "label": "Bible"},
        "Isaiah": {"color": "#3498DB", "icon": "✝️", "label": "Bible"},
        "Sira": {"color": "#F1C40F", "icon": "🐪", "label": "History"},         
        "Athanasian": {"color": "#9B59B6", "icon": "⚖️", "label": "Creed"}       
    }
    
    if response:
        with response_placeholder.container():
            st.markdown("### 🗣️ Agent's Apologetic Argument")
            st.markdown("---")
            st.write(response)
            
            # --- THE RETRACTABLE TOGGLE ---
            with st.expander("🕵️ View Forensic Source Trail", expanded=False):
                st.caption("These specific segments were injected into the AI's context window to ensure accuracy:")
                
                cols = st.columns(2) 
                for i, item in enumerate(candidates):
                    if item.get('selected', True):
                        source_name = item['source']
                        
                        # Default configuration
                        match_config = {"color": "#607D8B", "icon": "📄", "label": "Other"}
                        
                        # Apply specific styling if source matches
                        for key, config in source_map.items():
                            if key.lower() in source_name.lower():
                                match_config = config
                                break

                        with cols[i % 2]:
                            # Ensure distance is positive for display logic
                            confidence = max(0, (1 - item['distance'])) * 100
                            
                            st.markdown(
                                f"""
                                <div style="border-left: 5px solid {match_config['color']}; 
                                            padding: 8px; 
                                            margin-bottom: 8px; 
                                            background-color: rgba(128,128,128,0.1); 
                                            border-radius: 4px;
                                            font-size: 0.8rem;">
                                    <strong>{match_config['icon']} {source_name}</strong><br>
                                    <span style="color: {match_config['color']}; font-weight: bold;">
                                        Match Confidence: {confidence:.1f}%
                                    </span>
                                </div>
                                """, 
                                unsafe_allow_html=True
                            )

def parse_cwa_output(raw_output):
    """Robustly cleans and parses the Gemini's raw CWA output."""
    if not raw_output: return None
    cleaned_output = raw_output.strip()
    if cleaned_output.startswith("```json"):
        cleaned_output = cleaned_output.lstrip("```json").rstrip("```").strip()
    try:
        return json.loads(cleaned_output)
    except Exception as e:
        return {
            'weakest_points': f"Error parsing: {e}",
            'defense_strategy': f"Raw: {cleaned_output[:100]}..."
        }

def display_context_refinement():
    """Allows user selection of RAG segments."""
    run_id = st.session_state.get('current_run_uuid', 'init')
    
    # NO 'with ctx' block needed here
    st.markdown("### 📄 Retrieved Candidates & Refinement")
    candidates = st.session_state.get('retrieved_candidates', [])
    
    if not candidates:
        st.info("No segments retrieved yet.")
        return

    if st.button("Select/Deselect All", key=f"btn_toggle_{run_id}"): 
        current_state = all(item.get('selected', True) for item in candidates)
        for item in candidates: 
            item['selected'] = not current_state
        st.rerun()

    for i, item in enumerate(candidates):
        item['selected'] = st.checkbox(
            f"**{i+1}. {item['source']}**",
            value=item.get('selected', True),
            key=f"cb_{run_id}_{i}" 
        )
        with st.expander(f"View Segment {i+1} Details"):
            st.write(item['text'])

def display_defense_analysis(placeholder):
    """Renders the analysis section with metrics and the export button."""
    with placeholder.container():
        st.header("📊 Defense Analysis & Run Metrics")
        
        # 1. Metrics Row
        m1, m2, m3 = st.columns(3)
        # Calculate dynamic score or use session state
        m1.metric("Context Utilization", "85%") 
        m2.metric("Total Tokens", st.session_state.get('last_total_tokens', 0))
        m3.metric("Response Time", f"{round(st.session_state.get('last_response_time', 0), 2)}s")

        # 2. CWA Display Logic
        cwa = st.session_state.get('weakness_assessment')
        if st.session_state.get('last_ran_stance') != "Didactic/Explanatory":
            if cwa:
                st.subheader("🛡️ Proactive Defense Strategy")
                st.info(f"**Vulnerability:** {cwa.get('weakest_points')}")
                st.success(f"**Counter-Strategy:** {cwa.get('defense_strategy')}")
        else:
            st.info("Analysis section skipped in Didactic/Explanatory Mode.")

                # --- 5. CROSS-EXAMINATION LOGIC ---
def display_cross_examination():
    """Handles the follow-up debate logic."""
    if st.session_state.get('last_response'):
        st.markdown("---")
        st.markdown("### ⚔️ Cross-Examine the Agent")
        
        with st.form(key="cross_exam_form"):
            follow_up = st.text_input("Challenge the argument or ask a follow-up:")
            submit_follow_up = st.form_submit_button("Submit Challenge")

        if submit_follow_up and follow_up:
            with st.status("🕵️ Analyzing challenge..."):
                # Use current retrieval logic to get fresh context for the challenge
                new_context_data = retrieve_segments(follow_up)
                new_context_str = "\n".join([d['doc'] for d in new_context_data[:5]])
                
                llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.7)
                prompt = f"""
                You are the LexiCore Agent in {st.session_state['last_ran_stance']} mode.
                PREVIOUS ARGUMENT: {st.session_state['last_response']}
                NEW CHALLENGE: {follow_up}
                ADDITIONAL CONTEXT: {new_context_str}
                INSTRUCTION: Defend your position firmly. Stay in character.
                """
                response = llm.invoke(prompt)
                
                # Update history for the export
                if 'history' not in st.session_state: st.session_state['history'] = []
                st.session_state['history'].append({"role": "User Challenge", "content": follow_up})
                st.session_state['history'].append({"role": "Agent Defense", "content": response.content})
                
                st.session_state['last_response'] = response.content
                st.rerun()

def get_source_distribution():
    """Calculates the frequency of different sources in the current retrieval set."""
    candidates = st.session_state.get('retrieved_candidates', [])
    if not candidates:
        return {}, {}

    source_counts = {}
    category_counts = {"Christian": 0, "Islamic": 0, "Historical/Other": 0}

    for item in candidates:
        if item['selected']:
            # Count by specific source name (e.g., "John")
            src = item['source'].split()[0].replace(':', '')
            source_counts[src] = source_counts.get(src, 0) + 1
            
            # Count by Category using your existing classify_source function
            cat = classify_source(item['source'])
            category_counts[cat] += 1
            
    return source_counts, category_counts

def display_topic_heatmap():
    """Displays a visual breakdown of where the evidence is coming from."""
    if st.session_state.get('retrieved_candidates'):
        source_counts, category_counts = get_source_distribution()
        
        st.markdown("### 🗺️ Evidence Heatmap")
        
        # 1. Category Breakdown
        cols = st.columns(3)
        total = sum(category_counts.values())
        if total > 0:
            with cols[0]:
                st.write(f"✝️ **Christian**: {category_counts['Christian']}")
                st.progress(category_counts['Christian'] / total)
            with cols[1]:
                st.write(f"🌙 **Islamic**: {category_counts['Islamic']}")
                st.progress(category_counts['Islamic'] / total)
            with cols[2]:
                st.write(f"📜 **History**: {category_counts['Historical/Other']}")
                st.progress(category_counts['Historical/Other'] / total)

        # 2. Specific Sources Heatmap (Bar Chart)
        if source_counts:
            st.markdown("#### Specific Source Density")
            st.bar_chart(source_counts)
        else:
            st.info("No source data available for charting.")

            # --- UI COMPONENTS ---

def display_export_options():
    """Provides a button to download the current response as a professional PDF."""
    if st.session_state.get('last_response'):
        st.markdown("---")
        st.markdown("### 💾 Save This Session")
        
        # 1. BUILD TRANSCRIPT (Updated to ensure query is included)
        mode = st.session_state.get('last_ran_stance', 'Didactic')
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        
        # FIX: Try to get the query from multiple possible state locations
        query = st.session_state.get('last_query_input') or st.session_state.get('query_main') or "N/A"
        
        transcript = f"THEOLOGICAL INQUIRY: {query}\n" # Changed label for clarity
        transcript += f"Operation Mode: {mode}\n"
        transcript += f"Generated On: {timestamp}\n"
        transcript += "-"*40 + "\n\n"
        transcript += f"AGENT RESPONSE:\n{st.session_state.get('last_response')}\n\n"
        
        # Include cross-examination history if it exists
        if 'history' in st.session_state and st.session_state['history']:
            transcript += "CROSS-EXAMINATION HISTORY:\n"
            for entry in st.session_state['history']:
                transcript += f"{entry['role']}: {entry['content']}\n\n"

        # 2. GENERATE PDF
        try:
            candidates = st.session_state.get('retrieved_candidates', [])
            # Convert bytearray to bytes for Streamlit compatibility
            pdf_bytes = bytes(generate_pdf(transcript, sources=candidates))
            
            st.download_button(
                label="📥 Download Forensic PDF",
                data=pdf_bytes,
                file_name=f"lexicore_report_{uuid.uuid4().hex[:8]}.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.error(f"Error generating PDF: {e}")

# --- 5. MAIN APP LAYOUT & LOGIC ---

def set_custom_background(image_url, opacity=0.85):
    """
    Sets the background image and adjusts the transparency 
    of content cards to ensure readability.
    """
    if not image_url:
        # Fallback to dark theme gradient if no image is selected
        st.markdown("""
            <style>
            .stApp {
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            
            /* Apply Glassmorphism to the main content area */
            [data-testid="stVerticalBlock"] > div:has(div.stMarkdown) {{
                background-color: rgba(15, 23, 42, {opacity}) !important;
                backdrop-filter: blur(10px);
                padding: 25px;
                border-radius: 20px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }}
            </style>
            """,
            unsafe_allow_html=True
        )

import time

async def receive_and_play_audio(session):
    async for response in session.receive():
        if response.server_content and response.server_content.model_turn:
            for part in response.server_content.model_turn.parts:
                if part.inline_data:
                    audio_base64 = base64.b64encode(part.inline_data.data).decode()
                    st.session_state["latest_audio"] = audio_base64

def get_live_config():
    """Configures the live debating persona."""
    return {
        "response_modalities": ["AUDIO"],
        "speech_config": {
            "voice_config": {"prebuilt_voice_config": {"voice_name": "Kore"}} 
        },
        "system_instruction": f"""
        You are the LexiCore Forensic Debater. You are in a LIVE voice debate.
        Stance: {st.session_state.get('debate_stance', 'Scholarly')}
        Instruction: Listen to the user's vocal arguments and respond immediately. 
        Be firm, logical, and use the theological context provided.
        """
    }

# 2. CREATE A FRAGMENT FOR THE METER
# This updates ONLY this section every 0.1 seconds without rerunning the whole app
@st.fragment(run_every=0.1)
def volume_meter():
    current_vol = 0.0
    try:
        while not st.session_state.vol_queue.empty():
            current_vol = st.session_state.vol_queue.get_nowait()
    except:
        pass
    st.progress(current_vol, text=f"Mic Input Level: {int(current_vol * 100)}%")

if "vol_queue" not in st.session_state:
    st.session_state.vol_queue = queue.Queue()
if "ws_client" not in st.session_state:
    st.session_state.ws_client = None

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

@st.fragment(run_every=0.5)
def audio_receiver():
    if st.session_state.get("ws_client"):
        try:
            # We use a very short timeout to keep the UI snappy
            st.session_state.ws_client.settimeout(0.01)
            audio_b64 = st.session_state.ws_client.recv()
            
            if audio_b64 and isinstance(audio_b64, str):
                import time
                ts = time.time()
                # Unique ID forces the browser to treat this as a new sound
                st.markdown(
                    f'<audio autoplay src="data:audio/wav;base64,{audio_b64}" id="{ts}"></audio>', 
                    unsafe_allow_html=True
                )
        except:
            pass

def display_voice_chamber():
    st.divider()
    st.subheader("🎙️ Live Forensic Chamber")

    # 1. Initialize session states
    if "vol_queue" not in st.session_state:
        st.session_state.vol_queue = queue.Queue()
    
    # 2. Connection Logic (using existing websocket logic)
    if st.button("🎙️ Initialize LexiCore"):
        try:
            ws = websocket.create_connection("ws://localhost:8000/voice", timeout=5)
            st.session_state.ws_client = ws
            st.success("LexiCore Online")
        except Exception as e:
            st.error(f"Engine Offline: {e}")

    # 3. The Volume Meter (Visual Feedback)
    current_vol = 0.0
    while not st.session_state.vol_queue.empty():
        try: 
            current_vol = st.session_state.vol_queue.get_nowait()
        except: 
            break
    st.progress(current_vol, text=f"Mic Input Level: {int(current_vol * 100)}%")

    # 4. Define the Audio Callback
    def audio_callback(frame: av.AudioFrame):
        try:
            audio_data = frame.to_ndarray()
            # Update Volume Meter
            v_max = np.max(np.abs(audio_data))
            st.session_state.vol_queue.put(float(min(v_max / 25000, 1.0)))
            
            # Resample and Send via WebSocket
            if len(audio_data.shape) > 1: audio_data = audio_data[0]
            pcm_data = audio_data[::3].astype('int16').tobytes()
            
            if st.session_state.ws_client:
                st.session_state.ws_client.send_binary(pcm_data)
        except Exception:
            pass 
        return frame

    # 5. Capture the current thread context
    ctx = st.runtime.scriptrunner.get_script_run_ctx()

    # 6. The Audio Streamer
    # CHANGED: 'rtc_configuration' now points to 'RTC_CONFIG' defined on line 741
    webrtc_ctx = webrtc_streamer(
        key="voice-mic-stream",
        mode=WebRtcMode.SENDONLY,
        audio_frame_callback=audio_callback,
        rtc_configuration=RTC_CONFIG, 
        media_stream_constraints={"audio": True, "video": False},
        async_processing=True, 
    )

    # 7. Bridge the Thread Context
    if webrtc_ctx.state.playing:
        # Find the WebRTC background thread by name
        for thread in threading.enumerate():
            if "async_media_processor" in thread.name:
                add_script_run_ctx(thread)

    # 8. Launch playback receiver
    audio_receiver()

def main():
    st.set_page_config(page_title="Armor of Truth", layout="wide")
    apply_ultra_style() 

    # --- STARTUP LOGIC ---
    if 'startup_finished' not in st.session_state:
        placeholder = st.empty()
        with placeholder.container():
            st.markdown('<div class="startup-wrapper"><h1>THE ARMOR</h1><p>CONNECTING...</p></div>', unsafe_allow_html=True)
            emb_model, coll = load_resources() 
            st.session_state.embedding_model = emb_model
            st.session_state.collection = coll
            time.sleep(0.5)
        placeholder.empty()
        st.session_state['startup_finished'] = True

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("⚙️ Settings")
        bg_key = st.selectbox("🖼️ Background", list(BACKGROUND_IMAGES.keys()), key='background_key')
        bg_opacity = st.slider("🌫️ Box Opacity", 0.0, 1.0, 0.85, key='opacity_setting')
        set_custom_background(BACKGROUND_IMAGES.get(bg_key, ""), bg_opacity)
        
        st.divider()
        st.radio("📚 Filter", ["Textual (All)", "Scripture Only", "Historical Only"], key='source_filter')
        st.slider("🧠 Temperature", 0.0, 1.0, key='llm_temperature')
        st.radio("🗣️ Stance", ["Didactic/Explanatory", "Scholarly (Debate)", "Skeptical/Contrarian"], key='debate_stance')
        st.selectbox("📝 Citations", ["Inline (Scholarly)", "Minimalist", "Detailed Footnote Style"], key='citation_mode')

    # --- UI LAYOUT ---
    st.title("🛡️ Armor Of Truth")
    enable_voice = st.toggle("🔊 Enter Live Voice Chamber", key="enable_voice_toggle")

    if enable_voice:
        display_voice_chamber()
    else:
        # Standard Text Interface
        user_query = st.text_input("Enter the concept or challenge:", 
                                  value=st.session_state.get('query_input_text', ""),
                                  key="query_main")
        
        if st.button("Generate Apologetic Defense", type="primary"):
            st.session_state.run_generation = True
            st.session_state['last_query_input'] = user_query
            st.rerun()

        # Define tabs and placeholders to avoid "not defined" errors
        analysis_tab, response_tab, context_tab = st.tabs(["📊 Analysis", "🗣️ Response", "📄 Context"])
        with analysis_tab: analysis_placeholder = st.empty()
        with response_tab: response_placeholder = st.empty()
    
    # --- GENERATION LOGIC BLOCK ---
    if st.session_state.get('run_generation', False) and st.session_state.get('debate_result') is None:
        is_debate_mode = st.session_state.get('last_ran_stance') in ["Scholarly (Debate)", "Skeptical/Contrarian"]
        
        with st.status("🛡️ Processing...", expanded=True) as status:
            try:
                current_query = st.session_state['last_query_input']
                challenge_text = ""
                counter_strategy_text = ""

                # 1. Query Expansion
                if st.session_state.get('refine_query_enabled', False):
                    status.write("🔍 Refining query via adversarial expansion...")
                    aqe_chain = create_adversarial_query_expander(st.session_state['llm_temperature'])
                    expansion = aqe_chain.invoke({'query': current_query})
                    current_query = f"{current_query}. Context: {expansion}"

                # 2. Retrieval with Source Filtering
                # status.write("📚 Accessing scriptural database...")
                
                # Fetch all potential matches first
                raw_results = retrieve_segments(current_query)
                
                if not raw_results:
                    st.warning("No relevant segments found. Check your database.")
                    st.stop()

                # Get the active filter from sidebar
                active_filter = st.session_state.get('source_filter', 'Textual (All)')

                retrieved_context_texts = []
                for item in raw_results:
                    source_name = item['meta'].get('scripture_source', 'Unknown')
                    
                    # --- FILTER LOGIC ---
                    # Only add to context if it matches the sidebar selection
                    if active_filter == "Scripture Only":
                        # Only allow Bible or Church Fathers
                        if not any(x in source_name for x in ["Bible", "Orthodox", "Clement", "Ignatius"]):
                            continue
                    
                    elif active_filter == "Historical Only":
                        # Only allow secular/historical sources
                        if any(x in source_name for x in ["Quran", "Bible", "Surah"]):
                            continue
                    
                    # Add to the candidates list
                    retrieved_context_texts.append({
                        "id": item['id'], 
                        "text": item['doc'],
                        "source": source_name,
                        "distance": item['distance'], 
                        "selected": True, 
                    })

                # Safety check: If filter was too strict, fallback to top 5 raw results
                if not retrieved_context_texts:
                    retrieved_context_texts = [{"id": r['id'], "text": r['doc'], "source": r['meta'].get('scripture_source', 'N/A'), "distance": r['distance'], "selected": True} for r in raw_results[:5]]

                # Limit to top 20 filtered results and store
                st.session_state['retrieved_candidates'] = retrieved_context_texts[:20]
                final_context_string = get_context_for_llm()

                # 3. Critical Weakness Assessment (CWA)
                if is_debate_mode:
                    # status.write("⚖️ Analyzing theological vulnerabilities...")
                    initial_chain = create_debate_chain(retrieve_segments, st.session_state['llm_temperature'], st.session_state['last_ran_stance'], st.session_state['citation_mode'])
                    resp_temp = initial_chain.invoke({"context": final_context_string, "query": current_query, "challenge": "", "counter_strategy": ""})
                    
                    try:
                        assessor = create_weakness_assessor(st.session_state['llm_temperature'])
                        raw_cwa = assessor.invoke({"argument": resp_temp, "context": final_context_string})
        
                        # Guard against None or bad parsing
                        if raw_cwa and isinstance(raw_cwa, dict):
                            st.session_state['weakness_assessment'] = raw_cwa
                        else:
                            raise ValueError("Invalid JSON format from AI")
            
                    except Exception as e:
                        # FALLBACK: Prevents the 'NoneType' has no attribute 'get' error
                        st.session_state['weakness_assessment'] = {
                            "weakest_points": "The adversary was unable to find a specific weakness.",
                            "defense_strategy": "Maintain standard scriptural defense."
                        }
    
                    challenge_text = st.session_state['weakness_assessment'].get('weakest_points', "")
                    counter_strategy_text = st.session_state['weakness_assessment'].get('defense_strategy', "")
                # 4. Final Response Generation
                # status.write("✨ Formulating final response...")
                start_time = time.time()
                final_chain = create_debate_chain(retrieve_segments, st.session_state['llm_temperature'], st.session_state['last_ran_stance'], st.session_state['citation_mode'])
                response_final = final_chain.invoke({
                    "context": final_context_string, 
                    "query": st.session_state['last_query_input'],
                    "challenge": challenge_text, 
                    "counter_strategy": counter_strategy_text
                })
                
                # Store Final Results
                st.session_state['last_response'] = response_final
                st.session_state['last_response_time'] = time.time() - start_time
                st.session_state['last_total_tokens'] = int((len(final_context_string) + len(response_final)) / 4)
                
                st.session_state['debate_result'] = "Success"
                st.session_state.run_generation = False 
                status.update(label="✅ Generation Complete!", state="complete", expanded=False)
                st.rerun() 

            except Exception as e:
                status.update(label="❌ Error Occurred", state="error")
                st.error(f"Critical System Error: {e}")
                st.session_state['debate_result'] = "Error"
                st.session_state.run_generation = False

    # --- PERSISTED TAB CONTENT ---
    if st.session_state.get('last_response'):
        with response_tab:
            # We don't pass 'st' anymore
            display_agent_response(response_placeholder)
            display_cross_examination()
            display_export_options()

        with analysis_tab:
            # Placeholder is used for specific positioning
            display_defense_analysis(analysis_placeholder)
            display_topic_heatmap()

        with context_tab:
            # Simply call it; it will render inside context_tab automatically
            display_context_refinement()
if __name__ == "__main__":
    # Initialize all states globally
    initial_states = {
        'db_segments': 0, 
        'retrieved_candidates': [], 
        'last_response': "", 
        'source_filter': 'Textual (All)', 
        'llm_temperature': 0.5, 
        'debate_stance': "Didactic/Explanatory", 
        'current_run_uuid': str(uuid.uuid4())[:8],
        'run_generation': False,
        'debate_result': None,
        'last_ran_stance': "Didactic/Explanatory",
        'citation_mode': "Inline (Scholarly)",
        'weakness_assessment': None,
        'last_query_input': ""
    }
    
    for key, val in initial_states.items():
        if key not in st.session_state: 
            st.session_state[key] = val
            
    main()