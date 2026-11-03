import os
import tempfile

import faiss
import numpy as np
import streamlit as st
from Evtx.Evtx import Evtx
from sentence_transformers import SentenceTransformer

from llm import explain_answer


# ---------- Page setup ----------
st.set_page_config(
    page_title="AI-SOC Event Log Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.title("AI-SOC + Windows Event Log Q&A System")
st.write("Upload a Windows `.evtx` file and ask questions about its events.")


# ---------- Load embedding model ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


embedding_model = load_embedding_model()


# ---------- Extract text from EVTX ----------
def extract_evtx_text(evtx_file):
    """Save the uploaded EVTX temporarily and extract its event records as text."""
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".evtx",
            delete=False,
        ) as temp_file:
            temp_file.write(evtx_file.getvalue())
            temp_path = temp_file.name

        event_texts = []

        with Evtx(temp_path) as event_log:
            for record in event_log.records():
                # Each record is XML. Keep it as readable, searchable text.
                xml_text = record.xml()
                event_texts.append(" ".join(xml_text.split()))

        return "\n".join(event_texts)

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ---------- Split extracted text into chunks ----------
def chunk_text(text, chunk_size=1000, overlap=150):
    """Split text into overlapping chunks for retrieval."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# ---------- Create FAISS vector store ----------
def create_vector_store(chunks):
    """Embed chunks and store the vectors in a FAISS index."""
    if not chunks:
        return None

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Normalize vectors so inner-product search behaves like cosine similarity.
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index, embeddings


# ---------- Retrieve relevant chunks ----------
def retrieve_relevant_chunks(question, chunks, index, top_k=4):
    if index is None or not chunks:
        return []

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
    )
    question_embedding = np.asarray(question_embedding, dtype=np.float32)
    faiss.normalize_L2(question_embedding)

    k = min(top_k, len(chunks))
    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append(
                {
                    "text": chunks[idx],
                    "score": float(score),
                }
            )

    return results


# ---------- Upload and process EVTX ----------
evtx_file = st.file_uploader(
    "Upload Windows Event Log",
    type=["evtx"],
)

if evtx_file:
    with st.spinner("Reading the Windows event log..."):
        try:
            extracted_text = extract_evtx_text(evtx_file)

            if not extracted_text.strip():
                st.warning("No event records were found in this file.")
                st.stop()

            chunks = chunk_text(extracted_text)
            vector_store = create_vector_store(chunks)

            if vector_store is None:
                st.warning("Could not create a searchable index from this file.")
                st.stop()

            index, _ = vector_store

        except Exception as error:
            st.error(f"Could not process this EVTX file: {error}")
            st.stop()

    st.success(f"Processed the event log into {len(chunks)} searchable chunks.")

    with st.expander("Preview extracted event text"):
        st.text(extracted_text[:5000])

    # ---------- Ask questions ----------
    question = st.text_input(
        "Ask a question about the event log",
        placeholder="Example: What happened before PowerShell was executed?",
    )

    if st.button("Analyze", type="primary", disabled=not question.strip()):
        with st.spinner("Searching the event log and preparing an answer..."):
            relevant_chunks = retrieve_relevant_chunks(
                question,
                chunks,
                index,
            )

            if not relevant_chunks:
                st.warning("No relevant event records were found.")
            else:
                context = "\n\n".join(
                    item["text"] for item in relevant_chunks
                )

                try:
                    answer = explain_answer(question, context)
                    st.subheader("Analysis")
                    st.write(answer)

                    with st.expander("Relevant event records"):
                        for number, item in enumerate(relevant_chunks, start=1):
                            st.markdown(f"**Record {number}**")
                            st.code(item["text"], language="xml")

                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")
