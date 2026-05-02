import streamlit as st
from llm import explain_answer
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# -------------------------------
# Load embedding model
# -------------------------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------------------------------
# Extract text from PDF
# -------------------------------
def extract_pdf_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


# -------------------------------
# Split text into chunks
# -------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks


# -------------------------------
# Create FAISS vector store
# -------------------------------
def create_vector_store(chunks):
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index, embeddings


# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, chunks, index, k=3):
    q_embedding = embed_model.encode([query])
    D, I = index.search(np.array(q_embedding), k)
    return [chunks[i] for i in I[0]]


# -------------------------------
# UI
# -------------------------------
st.title("🛡️ AI-SOC + PDF Q&A System")

# Upload PDF
pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

query = st.text_area("Ask question:")

if pdf_file:
    text = extract_pdf_text(pdf_file)
    chunks = chunk_text(text)
    index, embeddings = create_vector_store(chunks)

    if st.button("🔍 Ask"):
        if not query.strip():
            st.warning("Enter a question")
        else:
            with st.spinner("Analyzing PDF..."):
                docs = retrieve(query, chunks, index)

                context = "\n\n".join(docs)

                # Combine context + query
                final_query = f"""
Use the below context to answer:

{context}

Question:
{query}
"""

                answer = explain_answer(final_query, "Answer based on document")

            st.subheader("📄 Answer")
            st.success(answer)