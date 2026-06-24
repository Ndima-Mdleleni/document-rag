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

def extract_text(uploaded_file):
    if uploaded_file.type == "application/pdf":
        import fitz
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        # clean up common PDF artifacts
        text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
        return text
    else:
        return uploaded_file.read().decode("utf-8")

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

def answer_question(question, chunks, chunk_embeddings, conversation_history=[]):
    relevant = get_relevant_chunks(question, chunks, chunk_embeddings)
    context  = "\n\n".join([f"[Source {i+1}]: {chunk}" for i, chunk in enumerate(relevant)])

    messages = conversation_history + [{
        "role": "user",
        "content": f"""Answer the question based only on the context below.
If the answer is not in the context, say so clearly.
Always end your answer with: Sources: [list which Source numbers you used]

Context:
{context}

Question: {question}"""
    }]

    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            messages=messages
        )
        return message.content[0].text, relevant
    except Exception as e:
        return f"Error: {str(e)}", []

uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf"])

if uploaded_file:
    text   = extract_text(uploaded_file)
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

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]  # exclude the question just added
                ]
                answer, sources = answer_question(question, chunks, chunk_embeddings, history)
            st.markdown(answer)
            if sources:
                with st.expander("📄 View sources used"):
                    for i, source in enumerate(sources):
                        clean = " ".join(source.split())
                        st.caption(f"**Source {i+1}:**")
                        st.caption(clean[:300] + "..." if len(clean) > 300 else clean)
                        st.divider()
            st.session_state.messages.append({"role": "assistant", "content": answer}) 
else:
    st.info("Upload a document to get started")
