import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Agentic RAG Explorer", layout="wide")
st.title("🤖 Agentic RAG System (CRAG)")

if "thread_id" not in st.session_state:            # NEW
    st.session_state.thread_id = None

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

    st.divider()                                     # NEW
    if st.button("🔄 New Conversation"):              # NEW
        st.session_state.thread_id = None
        st.rerun()

question = st.text_input("Ask a question about your documents:")
if st.button("Submit Question") and question:
    with st.spinner("Running LangGraph workflow..."):
        payload = {"question": question}
        if st.session_state.thread_id:                # NEW
            payload["thread_id"] = st.session_state.thread_id

        res = requests.post(f"{API_URL}/query", json=payload)
        if res.status_code == 200:
            data = res.json()
            st.session_state.thread_id = data["thread_id"]   # NEW

            st.markdown("### Answer")
            st.write(data["answer"])

            if data.get("used_web_search"):
                st.info("🌐 No relevant content found in your documents — this answer used a live web search instead.")

            if not data.get("grounded", True):
                st.warning("⚠️ This answer could not be fully verified against the source documents after retrying — treat it with caution.")

            with st.expander(f"🔍 Citations & Retrieved Chunks (Retries: {data['retry_count']})"):
                for idx, src in enumerate(data["sources"]):
                    st.markdown(f"**Chunk {idx+1}** (Score: `{src.get('score', 0):.3f}` | Page {src['metadata'].get('page')})")
                    st.info(src["text"])
        else:
            st.error("Error communicating with backend API.")