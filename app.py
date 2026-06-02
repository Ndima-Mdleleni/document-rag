import os
import numpy as np
import anthropic
import streamlit as st
from sentence_transformers import SentenceTransformer

# ── PAGE CONFIG ──────────────────────────────────────
st.set_page_config(
    page_title="Document RAG",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Document RAG System")
st.caption("Upload a document and ask questions about it")

# ── LOAD MODELS ──────────────────────────────────────
@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # try secrets first, then environment variable
    api_key = None
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
        st.write("key source: streamlit secrets")
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        st.write("key source: environment variable")
    
    if not api_key:
        st.error("No API key found")
        st.stop()
    
    st.write(f"key preview: {api_key[:10]}...")
    client = anthropic.Anthropic(api_key=api_key)
    return embedding_model, client

embedding_model, client = load_models()
