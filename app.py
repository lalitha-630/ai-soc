# -------------------------------
# Streamlit UI
# -------------------------------
st.title("AI-SOC PCAP Analysis + RAG")

pcap_file = st.file_uploader(
    "Upload a PCAP file",
    type=["pcap", "pcapng"]
)

query = st.text_area("Ask a question about the network traffic:")

if st.button("Analyze PCAP and answer"):
    if pcap_file is None:
        st.warning("Upload a PCAP file first.")
    elif not query.strip():
        st.warning("Enter a question first.")
    else:
        with st.spinner("Analyzing PCAP and generating answer..."):
            try:
                pcap_text = extract_pcap_text(pcap_file)

                if not pcap_text.strip():
                    st.error("No readable packet data was found.")
                    st.stop()

                chunks = chunk_text(pcap_text)

                if not chunks:
                    st.error("Could not create searchable chunks from this PCAP.")
                    st.stop()

                index = create_vector_store(chunks)
                docs = retrieve(query, chunks, index)
                context = "\n\n".join(docs)

                final_query = f"""
Use the packet information below to answer the question.
If there is not enough evidence, say so.
Do not call traffic malicious based only on a port or protocol.

Packet information:
{context}

Question:
{query}
"""

                answer = explain_answer(
                    final_query,
                    "Answer using the supplied PCAP packet information."
                )

                st.subheader("Analysis")

                if answer and str(answer).strip():
                    st.write(answer)
                else:
                    st.error(
                        "The LLM returned an empty answer. "
                        "Check that explain_answer() in llm.py returns the generated text."
                    )

            except Exception as e:
                st.error(f"Analysis failed: {e}")
