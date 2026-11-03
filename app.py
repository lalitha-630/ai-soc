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
    "Upload a PCAP file to summarize its packet-level network activity "
    "and request a SOC-style explanation."
)


def extract_pcap_context(uploaded_file, max_packets=10000):
    """Extract readable packet metadata from a PCAP file."""
    packet_lines = []
    protocol_counts = Counter()
    packet_count = 0

    # Scapy's PcapReader accepts a file-like object.
    file_bytes = io.BytesIO(uploaded_file.getvalue())

    try:
        with PcapReader(file_bytes) as reader:
            for packet in reader:
                packet_count += 1

                if packet_count > max_packets:
                    packet_lines.append(
                        f"Packet limit reached: analyzed first {max_packets} packets."
                    )
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
        return "No packets were found in the uploaded PCAP.", packet_count, protocol_counts

    summary = [
        f"Packets analyzed: {min(packet_count, max_packets)}",
        "Protocol counts: "
        + ", ".join(
            f"{protocol}={count}"
            for protocol, count in protocol_counts.most_common()
        ),
        "",
        "Packet details:",
        *packet_lines,
    ]

    return "\n".join(summary), packet_count, protocol_counts


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
                context, packet_count, protocol_counts = extract_pcap_context(
                    uploaded_file
                )
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

        st.subheader("Packet Summary")
        st.write(f"Packets read: **{packet_count}**")

        if protocol_counts:
            st.write(
                "Protocol counts: "
                + ", ".join(
                    f"{protocol}: {count}"
                    for protocol, count in protocol_counts.most_common()
                )
            )

        with st.expander("View extracted packet details"):
            st.text(context[:20000])
            if len(context) > 20000:
                st.caption("Display truncated; the full extracted context was sent for analysis.")

        with st.spinner("Generating SOC analysis..."):
            answer = explain_answer(
                query=question.strip()
                or "Summarize this traffic and identify anything suspicious.",
                base_answer=context[:30000],
            )

        st.subheader("Analysis")

        if answer.startswith("LLM_ERROR:"):
            st.error(answer)
            st.info(
                "Check that GROQ_API_KEY is configured in your local .env "
                "file or Streamlit Cloud Secrets, and confirm that the selected "
                "Groq model is available to your account."
            )
        else:
            st.write(answer)

else:
    st.info("Upload a PCAP or PCAPNG file to begin.")
