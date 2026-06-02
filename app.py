import os
import numpy as np
import anthropic
import streamlit as st
from sentence_transformers import SentenceTransformer

st.set_page_config(
    page_title="Document RAG",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Document RAG System")
st.caption("Upload a document and ask questions about it")

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    client = anthropic.Anthropic(api_key=api_key)
    return embedding_model, client

embedding_model, client = load_models()

def chunk_document(text, chunk_size=200, overlap=50):
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def get_relevant_chunks(question, chunks, chunk_embeddings, top_k=2):
    question_embedding = embedding_model.encode([question])[0]
    similarities = []
    for chunk_embedding in chunk_embeddings:
        similarity = np.dot(question_embedding, chunk_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append(similarity)
    top_indices     = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def answer_question(question, chunks, chunk_embeddings):
    relevant = get_relevant_chunks(question, chunks, chunk_embeddings)
    context  = "\n\n".join(relevant)
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": f"Answer this question based only on the context below.\n\nContext:\n{context}\n\nQuestion: {question}"
            }]
        )
        return message.content[0].text
    except Exception as e:
        return f"Error: {str(e)}"

uploaded_file = st.file_uploader("Upload a text document", type=["txt"])

if uploaded_file:
    text   = uploaded_file.read().decode("utf-8")
    chunks = chunk_document(text)
    chunk_embeddings = embedding_model.encode(chunks)

    st.success(f"Document loaded: {len(text.split())} words, {len(chunks)} chunks")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Ask a question about your document"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("thinking..."):
                answer = answer_question(question, chunks, chunk_embeddings)
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("Upload a document to get started")
