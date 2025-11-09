# app.py
import streamlit as st
import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_DB_DIR, EMBEDDING_MODEL, CHAT_MODEL
from llm_client import call_local_llm

st.set_page_config(page_title="AI Library Search", layout="wide")
st.title("🔎 AI Library Search")

client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection(name="ai_library")

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL
)

query = st.text_input("Search your AI materials:", placeholder="e.g. 'LoRA fine-tuning', 'langchain pdf loader', 'Pandas join by date'")

top_k = st.sidebar.slider("Results", 1, 10, 5)
use_llm = st.sidebar.checkbox("Ask LLM to summarise results", value=True)

if query:
    q_emb = embedding_fn([query])[0]
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=top_k
    )

    hits = list(zip(
        res.get("documents", [[]])[0],
        res.get("metadatas", [[]])[0],
    ))

    st.subheader("Results")
    for doc, meta in hits:
        st.markdown(f"**Source:** `{meta.get('source_path')}`  \n{doc}")
        st.markdown("---")

    if use_llm and hits:
        combined = "\n".join([d for d, _ in hits])
        prompt = f"""You are helping me reuse my own AI study notes.
Here are the most relevant notes:

{combined}

User query: {query}

Write a short answer (max 150 words) and point me to the most relevant file paths."""
        answer = call_local_llm(CHAT_MODEL, prompt)
        st.subheader("LLM Answer")
        st.write(answer)
