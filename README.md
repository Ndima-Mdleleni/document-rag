# Document RAG System

A retrieval-augmented generation system that lets you chat with any text document.

## What it is
A RAG system that reads a document, finds the most relevant sections based on 
your question, and generates an answer using that context.

## How it works
1. **Load** — reads your document and splits it into chunks
2. **Embed** — turns each chunk into a vector using sentence transformers
3. **Retrieve** — finds the most relevant chunks by comparing vectors
4. **Generate** — answers your question using the relevant context

## How to run it
pip install sentence-transformers transformers torch numpy anthropic streamlit

Set your Anthropic API key:
export ANTHROPIC_API_KEY="your-key-here"

Run the web interface:
streamlit run app.py

Then open your browser and upload any text document to start chatting with it.

## Upgrading to Claude API
Replace the GPT2 generator with the Anthropic Claude API for 
significantly better answers. Requires an Anthropic API key from console.anthropic.com.

## What I learned
This project extended the concepts from my previous projects — embeddings, vectors 
and language models — and applied them at a larger, more practical scale.

Building a RAG system showed me how to combine different tools based on their 
strengths rather than relying on a single solution. Using HuggingFace for embeddings 
and GPT2 for generation taught me how real ML systems are assembled from 
specialised components.

I was also introduced to cosine similarity — a method for comparing vectors by 
meaning rather than exact value. This gave me a new perspective on how models 
find relevant information at scale, which is fundamentally different from the 
linear regression approach I learned earlier.

Working with HuggingFace models and understanding how API keys work were also 
new additions to my toolkit that I'll carry into future projects.

## Built with
- Python
- Sentence Transformers
- HuggingFace Transformers
- NumPy
## Live Demo
https://document-rag-k6nb6x894hz77psqyweqvb.streamlit.app
