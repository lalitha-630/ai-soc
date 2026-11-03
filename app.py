import io
from collections import Counter

import streamlit as st
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, DNS, DNSQR

from llm import explain_answer


st.set_page_config(
    page_title="AI-SOC Threat Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.title("AI-SOC Threat Analyzer")
st.write(
    "Upload a PCAP file to summarize packet-level network activity "
    "and request a SOC-style explanation."
)

# Keep the LLM request small enough to avoid Groq token-limit errors.
MAX_PACKETS_TO_READ = 10000
MAX_PACKET_LINES_FOR_LLM = 30
MAX_CONTEXT_CHARS_FOR_LLM = 5000


def extract_pcap_context(uploaded_file, max_packets=MAX_PACKETS_TO_READ):
    """Extract packet metadata and return a summary plus packet lines."""
    packet_lines = []
    protocol_counts = Counter()
    packet_count = 0

    file_bytes = io.BytesIO(uploaded_file.getvalue())

    try:
        with PcapReader(file_bytes) as reader:
            for packet in reader:
                packet_count += 1

                if packet_count > max_packets:
                    break

                src_ip = "Unknown"
                dst_ip = "Unknown"

                if IP in packet:
                    src_ip = packet[IP].src
                    dst_ip = packet[IP].dst
                elif IPv6 in packet:
                    src_ip = packet[IPv6].src
                    dst_ip = packet[IPv6].dst

                protocol = "Other"
                src_port = "-"
                dst_port = "-"
                tcp_flags = "-"

                if TCP in packet:
                    protocol = "TCP"
                    src_port = packet[TCP].sport
                    dst_port = packet[TCP].dport
                    tcp_flags = str(packet[TCP].flags)
                elif UDP in packet:
                    protocol = "UDP"
                    src_port = packet[UDP].sport
                    dst_port = packet[UDP].dport

                protocol_counts[protocol] += 1

                line = (
                    f"Packet {packet_count}: "
                    f"src={src_ip}:{src_port}, "
                    f"dst={dst_ip}:{dst_port}, "
                    f"protocol={protocol}, "
                    f"length={len(packet)} bytes, "
                    f"TCP_flags={tcp_flags}"
                )

                if DNS in packet and packet[DNS].qd and DNSQR in packet:
                    dns_query = packet[DNSQR].qname
                    if isinstance(dns_query, bytes):
                        dns_query = dns_query.decode(
                            "utf-8", errors="replace"
                        )
                    line += f", DNS_query={dns_query}"

                packet_lines.append(line)

    except Exception as exc:
        raise ValueError(f"Could not read this PCAP file: {exc}") from exc

    if not packet_lines:
        return {
            "packet_count": packet_count,
            "protocol_counts": protocol_counts,
            "packet_lines": [],
            "full_context": "No packets were found in the uploaded PCAP.",
            "llm_context": "No packets were found in the uploaded PCAP.",
        }

    summary_lines = [
        f"Packets analyzed: {len(packet_lines)}",
        "Protocol counts: "
        + ", ".join(
            f"{protocol}={count}"
            for protocol, count in protocol_counts.most_common()
        ),
    ]

    full_context = "\n".join(summary_lines + ["", "Packet details:"] + packet_lines)

    # Send only a short sample to the LLM, not the entire PCAP.
    llm_packet_sample = packet_lines[:MAX_PACKET_LINES_FOR_LLM]
    llm_context = "\n".join(
        summary_lines
        + [
            "",
            f"Sample: first {len(llm_packet_sample)} packet records",
            *llm_packet_sample,
        ]
    )
    llm_context = llm_context[:MAX_CONTEXT_CHARS_FOR_LLM]

    return {
        "packet_count": len(packet_lines),
        "protocol_counts": protocol_counts,
        "packet_lines": packet_lines,
        "full_context": full_context,
        "llm_context": llm_context,
    }


uploaded_file = st.file_uploader(
    "Upload a PCAP or PCAPNG file",
    type=["pcap", "pcapng"],
)

question = st.text_input(
    "Question",
    value="Summarize this traffic and identify anything suspicious.",
)

if uploaded_file is not None:
    st.write(f"Uploaded file: **{uploaded_file.name}**")

    if st.button("Analyze PCAP", type="primary"):
        with st.spinner("Reading packet information..."):
            try:
                result = extract_pcap_context(uploaded_file)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.subheader("Packet Summary")
        st.write(f"Packets analyzed: **{result['packet_count']}**")

        if result["protocol_counts"]:
            st.write(
                "Protocol counts: "
                + ", ".join(
                    f"{protocol}: {count}"
                    for protocol, count in result["protocol_counts"].most_common()
                )
            )

        with st.expander("View extracted packet details"):
            st.text(result["full_context"][:30000])
            if len(result["full_context"]) > 30000:
                st.caption("Display truncated.")

        with st.spinner("Generating SOC analysis..."):
            answer = explain_answer(
                query=question.strip()
                or "Summarize this traffic and identify anything suspicious.",
                base_answer=result["llm_context"],
            )

        st.subheader("Analysis")

        if answer.startswith("LLM_ERROR:"):
            st.error(answer)
            st.info(
                "Check that GROQ_API_KEY is configured in your .env file "
                "or Streamlit Cloud Secrets, and confirm that the selected "
                "Groq model is available to your account."
            )
        else:
            st.write(answer)

else:
    st.info("Upload a PCAP or PCAPNG file to begin.")
