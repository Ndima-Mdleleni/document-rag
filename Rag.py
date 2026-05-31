import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ── 1. LOAD DOCUMENT ─────────────────────────────────
def load_document(filepath):
    with open(filepath, 'r') as f:
        return f.read()

# ── 2. CHUNK DOCUMENT ────────────────────────────────
def chunk_document(text, chunk_size=200, overlap=50):
    words  = text.split()
    chunks = []
    i      = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# ── 3. EMBED CHUNKS ──────────────────────────────────
def embed_chunks(chunks, model):
    return model.encode(chunks)

# ── 4. RETRIEVAL ─────────────────────────────────────
def get_relevant_chunks(question, chunks, chunk_embeddings, embedding_model, top_k=2):
    question_embedding = embedding_model.encode([question])[0]
    similarities = []
    for chunk_embedding in chunk_embeddings:
        similarity = np.dot(question_embedding, chunk_embedding) / (
            np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
        )
        similarities.append(similarity)
    top_indices    = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    return relevant_chunks

# ── 5. GENERATION ────────────────────────────────────
def answer_question(question, chunks, chunk_embeddings, embedding_model, generator):
    relevant = get_relevant_chunks(question, chunks, chunk_embeddings, embedding_model)
    context  = "\n\n".join(relevant)
    prompt   = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    result   = generator(prompt, max_length=300, truncation=True, do_sample=False)
    full_text = result[0]['generated_text']
    return full_text.split("Answer:")[-1].strip()

# ── 6. MAIN ──────────────────────────────────────────
filepath = sys.argv[1] if len(sys.argv) > 1 else "Document.txt"

print("loading document...")
text   = load_document(filepath)
chunks = chunk_document(text)
print(f"document loaded: {len(text.split())} words, {len(chunks)} chunks")

print("loading embedding model...")
embedding_model  = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = embed_chunks(chunks, embedding_model)
print("embeddings ready")

print("loading generation model...")
generator = pipeline('text-generation', model='gpt2')

print("\n--- RAG System Ready ---")
print(f"document: {filepath}")
print("type 'quit' to exit\n")

while True:
    question = input("you: ")
    if question.lower() == 'quit':
        break
    answer = answer_question(question, chunks, chunk_embeddings, embedding_model, generator)
    print(f"\nassistant: {answer}\n")