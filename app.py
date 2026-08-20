from __future__ import annotations

import os, time, uuid
from pathlib import Path
from io import BytesIO
import streamlit as st

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from lexicore.store import EvidenceStore
from lexicore.llm import answer, assess

st.set_page_config(page_title="LexiCore", page_icon="🛡️", layout="wide")
DB = os.getenv("LEXICORE_DB_PATH", "./chroma_db")
COLLECTION = os.getenv("LEXICORE_COLLECTION", "lexicore_evidence_v3")

@st.cache_resource(show_spinner="Loading evidence index…")
def get_store():
    return EvidenceStore(DB, COLLECTION)

def generate_pdf(history: list, weakness_data) -> bytes:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle('ReportTitle', parent=styles['Heading1'], fontSize=18, leading=22, textColor="#1f2937", spaceAfter=10)
    qa_heading = ParagraphStyle('QAHeading', parent=styles['Heading2'], fontSize=11, leading=15, textColor="#3b82f6", spaceBefore=8, spaceAfter=3)
    body_style = ParagraphStyle('ReportBody', parent=styles['Normal'], fontSize=9.5, leading=13.5, textColor="#374151", spaceAfter=6)
    
    story = [
        Paragraph("🛡️ LexiCore Research Report", title_style),
        Paragraph("Evidence-Grounded Theological Research & Apologetics", ParagraphStyle('Sub', parent=body_style, fontSize=10, textColor="#6b7280", spaceAfter=15))
    ]
    
    for turn in reversed(history):
        story.append(Paragraph(f"<b>Query:</b> {turn['query']}", qa_heading))
        story.append(Paragraph(f"<b>Answer:</b><br/>{turn['answer']}", body_style))
        story.append(Spacer(1, 6))
        
    if weakness_data:
        story.append(Paragraph("<b>Adversarial Review / Weaknesses</b>", qa_heading))
        for wp in weakness_data.weakest_points:
            story.append(Paragraph(f"• {wp}", body_style))
            
    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()

def init():
    defaults = {
        "hits": [], 
        "query": "", 
        "answer": None, 
        "weakness": None, 
        "run_id": str(uuid.uuid4().hex),
        "history": [],  # Current active conversation thread
        "sessions": {}, # Saved history sessions dictionary
        "active_session_id": None,
        # Persistent setting defaults (Source filters set to empty list by default)
        "setting_stance": "Didactic/Explanatory",
        "setting_categories": [],
        "setting_n": 15,
        "setting_temp": 0.2
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)

def main():
    init()
    st.title("🛡️ LexiCore")
    st.caption("Evidence-grounded theological research, apologetics, and cross-examination")
    
    try:
        store = get_store()
    except Exception as e:
        st.error(f"Could not open the canonical evidence index: {e}")
        st.info(f"Set LEXICORE_DB_PATH and build the index with: python ingest.py --data ./data --db \"{DB}\"")
        st.stop()
        
    with st.sidebar:
        st.header("Research controls")
        
        # Persistent widgets tied to session state keys
        stance = st.selectbox(
            "Mode", 
            ["Didactic/Explanatory", "Scholarly (Debate)", "Skeptical/Contrarian"],
            key="setting_stance"
        )
        
        selected_categories = st.multiselect(
            "Source filters",
            [
                "Christian Scripture",
                "Islamic Scripture",
                "Islamic Hadith",
                "Islamic History",
                "Christian Creed",
                "Christian Patristic",
                "Jewish Scripture / Commentary",
                "Historical / Other",
            ],
            key="setting_categories"
        )
        
        n = st.slider("Evidence segments", 5, 30, key="setting_n")
        temp = st.slider("Generation temperature", 0.0, 1.0, key="setting_temp", step=0.05)
        
        st.divider()
        st.subheader("Saved Conversations")
        
        if st.button("➕ New Research Thread", use_container_width=True):
            if st.session_state.history:
                if not st.session_state.active_session_id:
                    st.session_state.active_session_id = str(uuid.uuid4().hex)
                st.session_state.sessions[st.session_state.active_session_id] = {
                    "title": st.session_state.history[0]["query"][:30] + "...",
                    "history": list(st.session_state.history)
                }
            st.session_state.history = []
            st.session_state.answer = None
            st.session_state.weakness = None
            st.session_state.active_session_id = None
            st.rerun()
            
        if st.session_state.sessions:
            for sess_id, sess_data in list(st.session_state.sessions.items()):
                if st.button(f"💬 {sess_data['title']}", key=f"load_{sess_id}", use_container_width=True):
                    st.session_state.history = list(sess_data["history"])
                    st.session_state.active_session_id = sess_id
                    st.session_state.answer = None
                    st.session_state.weakness = None
                    st.rerun()
        
        if st.button("🗑️ Clear All History", use_container_width=True):
            st.session_state.history = []
            st.session_state.sessions = {}
            st.session_state.active_session_id = None
            st.session_state.answer = None
            st.session_state.weakness = None
            st.success("History cleared.")
            st.rerun()
            
    q = st.text_area(
        "Question / claim / counter-question",
        value=st.session_state.query,
        height=120,
        placeholder="Ask a theological question or follow up with a counter-question…",
    )
    
    c1, c2 = st.columns(2)
    retrieve = c1.button("🔎 Retrieve evidence", use_container_width=True)
    generate = c2.button("🛡️ Generate answer", type="primary", use_container_width=True)
    
    if retrieve or generate:
        if not q.strip():
            st.warning("Enter a question first.")
            st.stop()
        st.session_state.query = q
        
        cats = selected_categories if selected_categories else None
        
        with st.spinner("Retrieving evidence…"):
            st.session_state.hits = store.search(q, n=n, categories=cats)
            st.session_state.run_id = uuid.uuid4().hex
            
    if not st.session_state.hits and (retrieve or generate):
        st.warning("No matching evidence was found.")
        st.stop()
        
    hits = st.session_state.hits
    if hits:
        selected = [h.to_record() for h in hits]

        if generate:
            if not selected:
                st.error("Select at least one evidence segment.")
                st.stop()
            with st.spinner("Generating evidence-grounded response…"):
                started = time.perf_counter()
                
                result, used = answer(
                    query=q, 
                    records=selected, 
                    stance=stance, 
                    temperature=temp, 
                    history=st.session_state.history
                )
                
                elapsed = time.perf_counter() - started
                st.session_state.answer = result
                st.session_state.weakness = None
                
                if stance != "Didactic/Explanatory":
                    st.session_state.weakness, _ = assess(result.answer, used, stance=stance, temperature=temp)
                
                st.session_state.elapsed = elapsed
                
                st.session_state.history.append({
                    "query": q,
                    "answer": result.answer
                })
                
                if not st.session_state.active_session_id:
                    st.session_state.active_session_id = str(uuid.uuid4().hex)
                st.session_state.sessions[st.session_state.active_session_id] = {
                    "title": st.session_state.history[0]["query"][:30] + "...",
                    "history": list(st.session_state.history)
                }
                
                st.session_state.query = ""
                st.rerun()

    # Display conversation history if available
    if st.session_state.get("history"):
        st.divider()
        st.subheader("Conversation thread")
        
        pdf_bytes = generate_pdf(st.session_state.history, st.session_state.get("weakness"))
        st.download_button(
            label="📥 Download Research Thread as PDF",
            data=pdf_bytes,
            file_name="lexicore_research_report.pdf",
            mime="application/pdf",
            use_container_width=True
        )
        
        for turn in reversed(st.session_state.history):
            with st.chat_message("user"):
                st.write(turn["query"])
            with st.chat_message("assistant"):
                st.write(turn["answer"])

    result = st.session_state.get("answer")
    if result:
        if result.reasoning:
            with st.expander("Model reasoning"):
                st.write(result.reasoning)
        if result.limitations:
            with st.expander("Limitations & constraints"):
                for x in result.limitations:
                    st.write(f"- {x}")
        
        st.caption(f"Generation time: {st.session_state.get('elapsed', 0):.2f}s. Similarity is a ranking signal, not a truth probability.")
        
        if st.session_state.get("weakness"):
            w = st.session_state.weakness
            with st.expander("Adversarial review"):
                st.markdown("**Weakest points**")
                for x in w.weakest_points:
                    st.write(f"- {x}")
                st.markdown("**Defense / qualification strategy**")
                for x in w.defense_strategy:
                    st.write(f"- {x}")
                if w.unsupported_claims:
                    st.markdown("**Unsupported claims**")
                    for x in w.unsupported_claims:
                        st.write(f"- {x}")

if __name__ == "__main__":
    main()