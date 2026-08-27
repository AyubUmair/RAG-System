import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Explorer", layout="wide")
st.title("🤖 Agentic RAG System (CRAG)")

# Sidebar Ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file and st.button("Index Document"):
        with st.spinner("Processing & embedding..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            res = requests.post(f"{API_URL}/ingest", files=files)
            if res.status_code == 200:
                st.success(f"Indexed {res.json()['chunks_indexed']} chunks!")
            else:
                st.error("Failed to index document.")

# Main Chat Interface
question = st.text_input("Ask a question about your documents:")
if st.button("Submit Question") and question:
    with st.spinner("Running LangGraph workflow..."):
        res = requests.post(f"{API_URL}/query", json={"question": question})
        if res.status_code == 200:
            data = res.json()
            st.markdown("### Answer")
            st.write(data["answer"])
            if data.get("used_web_search"):
                st.info("🌐 No relevant content found in your documents — this answer used a live web search instead.")
            if not data.get("grounded", True):
                st.warning("⚠️ This answer could not be fully verified against the source documents after retrying — treat it with caution.")
            
            with st.expander(f"🔍 Citations & Retrieved Chunks (Retries: {data['retry_count']})"):
                for idx, src in enumerate(data.get("sources", [])):
                    st.markdown(f"**Chunk {idx+1}** (Score: `{src.get('score', 0):.3f}` | Page {src.get('metadata', {}).get('page')})")
                    st.info(src["text"])
        else:
            try:
                error_detail = res.json().get("detail", res.text)
            except Exception:
                error_detail = res.text
            st.error(f"Backend Error ({res.status_code}): {error_detail}")