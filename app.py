import streamlit as st
from llm import explain_answer
from sentence_transformers import SentenceTransformer
from scapy.utils import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.dns import DNS, DNSQR
import faiss
import numpy as np
from io import BytesIO

# -------------------------------
# Load embedding model
# -------------------------------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embedding_model()


# -------------------------------
# Extract packet metadata from PCAP
# -------------------------------
def extract_pcap_text(pcap_file, max_packets=10000):
    packet_lines = []
    reader = PcapReader(BytesIO(pcap_file.getvalue()))

    try:
        for packet_number, packet in enumerate(reader):
            if packet_number >= max_packets:
                packet_lines.append(
                    f"Capture truncated after {max_packets} packets."
                )
                break

            src_ip = dst_ip = "Unknown"
            protocol = "Other"
            src_port = dst_port = "N/A"
            tcp_flags = "N/A"

            if IP in packet:
                src_ip = packet[IP].src
                dst_ip = packet[IP].dst
            elif IPv6 in packet:
                src_ip = packet[IPv6].src
                dst_ip = packet[IPv6].dst

            if TCP in packet:
                protocol = "TCP"
                src_port = packet[TCP].sport
                dst_port = packet[TCP].dport
                tcp_flags = str(packet[TCP].flags)
            elif UDP in packet:
                protocol = "UDP"
                src_port = packet[UDP].sport
                dst_port = packet[UDP].dport

            packet_info = (
                f"Packet {packet_number + 1}: "
                f"src={src_ip}:{src_port}, "
                f"dst={dst_ip}:{dst_port}, "
                f"protocol={protocol}, "
                f"length={len(packet)} bytes, "
                f"TCP_flags={tcp_flags}"
            )

            if DNS in packet and packet[DNS].qd and DNSQR in packet:
                dns_query = packet[DNSQR].qname.decode(
                    "utf-8", errors="replace"
                ).rstrip(".")
                packet_info += f", DNS_query={dns_query}"

            packet_lines.append(packet_info)

    finally:
        reader.close()

    return "\n".join(packet_lines)


# -------------------------------
# Split extracted text into chunks
# -------------------------------
def chunk_text(text, chunk_size=300):
    words = text.split()
    return [
        " ".join(words[i:i + chunk_size])
        for i in range(0, len(words), chunk_size)
    ]


# -------------------------------
# Create FAISS vector store
# -------------------------------
def create_vector_store(chunks):
    embeddings = embed_model.encode(
        chunks,
        convert_to_numpy=True
    ).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index


# -------------------------------
# Retrieve relevant chunks
# -------------------------------
def retrieve(query, chunks, index, k=3):
    query_embedding = embed_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    k = min(k, len(chunks))
    _, indices = index.search(query_embedding, k)

    return [
        chunks[i]
        for i in indices[0]
        if 0 <= i < len(chunks)
    ]


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI-SOC PCAP Analysis + RAG")

pcap_file = st.file_uploader(
    "Upload a PCAP file",
    type=["pcap", "pcapng"]
)

query = st.text_area("Ask a question about the network traffic:")

if pcap_file:
    if st.button("Analyze PCAP"):
        with st.spinner("Reading packets and preparing context..."):
            try:
                pcap_text = extract_pcap_text(pcap_file)

                if not pcap_text.strip():
                    st.error("No readable packets were found in this capture.")
                    st.stop()

                chunks = chunk_text(pcap_text)

                if not chunks:
                    st.error("Could not create searchable content from this capture.")
                    st.stop()

                index = create_vector_store(chunks)

                st.session_state["pcap_chunks"] = chunks
                st.session_state["pcap_index"] = index

                st.success(f"Processed {len(chunks)} text chunks from the PCAP.")

            except Exception as e:
                st.error(f"Could not read this PCAP file: {e}")

if "pcap_chunks" in st.session_state:
    if st.button("Ask about this PCAP"):
        if not query.strip():
            st.warning("Enter a question.")
        else:
            with st.spinner("Searching traffic and generating an explanation..."):
                docs = retrieve(
                    query,
                    st.session_state["pcap_chunks"],
                    st.session_state["pcap_index"]
                )

                context = "\n\n".join(docs)

                final_query = f"""
Answer the question using only the packet information below.
If the capture does not contain enough evidence, say so.
Do not claim that traffic is malicious based on a port or protocol alone.

Packet information:
{context}

Question:
{query}
"""

                answer = explain_answer(
                    final_query,
                    "Answer based on the supplied PCAP packet information."
                )

            st.subheader("Analysis")
            st.write(answer)
