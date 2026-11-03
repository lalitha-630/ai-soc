import io
import os
from collections import Counter

import gdown
import joblib
import pandas as pd
import streamlit as st
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, DNS, DNSQR

from llm import explain_answer


# -------------------- Page setup and styling --------------------

st.set_page_config(
    page_title="AI-SOC Threat Analyzer",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        background: #0b1220;
        color: #e5e7eb;
    }
    [data-testid="stSidebar"] {
        background: #111c30;
        border-right: 1px solid #263750;
    }
    [data-testid="stSidebar"] * {
        color: #e5e7eb;
    }
    h1, h2, h3 {
        color: #f8fafc;
    }
    [data-testid="stFileUploader"],
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        background: #17243a;
        color: #e5e7eb;
    }
    .stButton > button {
        width: 100%;
        background: #1e3a5f;
        color: white;
        border: 1px solid #345477;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: #28517f;
        color: white;
        border-color: #6d9acb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MAX_PACKETS_TO_READ = 10000
MAX_PACKET_LINES_FOR_LLM = 30
MAX_CONTEXT_CHARS_FOR_LLM = 5000

MODEL_PATH = "/tmp/model_bundle.joblib"
GOOGLE_DRIVE_FILE_ID = "1QSBGKdxnXwX1gzG1n_EO6rMAznARfDpy"


# -------------------- Sidebar navigation --------------------

if "page" not in st.session_state:
    st.session_state.page = "ML findings"

with st.sidebar:
    st.title("AI-SOC")
    st.caption("Threat Analyzer")
    st.divider()

    if st.button("ML findings", key="nav_ml"):
        st.session_state.page = "ML findings"

    if st.button("RAG investigation", key="nav_rag"):
        st.session_state.page = "RAG investigation"

    if st.button("About", key="nav_about"):
        st.session_state.page = "About"


# -------------------- Load ML model from Google Drive --------------------

@st.cache_resource
def load_model():
    """Download the model from Drive if needed, then load the model bundle."""
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"

        # Do not add fuzzy=True; some gdown versions do not support it.
        downloaded_path = gdown.download(
            url,
            output=MODEL_PATH,
            quiet=False,
        )

        if not downloaded_path or not os.path.exists(MODEL_PATH):
            raise RuntimeError(
                "Could not download the model from Google Drive. "
                "Check that the file is shared as 'Anyone with the link — Viewer'."
            )

    loaded = joblib.load(MODEL_PATH)

    if isinstance(loaded, dict):
        model = loaded.get("pipeline") or loaded.get("model")
        feature_columns = loaded.get("feature_columns")
        label_map = loaded.get("label_map", {})
    else:
        model = loaded
        feature_columns = getattr(loaded, "feature_names_in_", None)
        label_map = {}

    if model is None:
        raise ValueError(
            "The saved bundle must contain a model under 'pipeline' or 'model'."
        )

    if feature_columns is None:
        feature_columns = getattr(model, "feature_names_in_", None)

    if feature_columns is None:
        raise ValueError(
            "The model's expected feature columns were not found. "
            "Save them in the bundle as 'feature_columns'."
        )

    return model, list(feature_columns), label_map


# -------------------- PCAP feature extraction --------------------

def packet_ip_pair(packet):
    if IP in packet:
        return packet[IP].src, packet[IP].dst, packet[IP].ttl, packet[IP].ttl

    if IPv6 in packet:
        return packet[IPv6].src, packet[IPv6].dst, 0, 0

    return None


def extract_pcap_context(uploaded_file, max_packets=MAX_PACKETS_TO_READ):
    """Extract packet details and aggregate them into bidirectional flows."""
    flows = {}
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

                pair = packet_ip_pair(packet)
                if not pair:
                    continue

                src_ip, dst_ip, src_ttl, dst_ttl = pair
                timestamp = float(packet.time)
                packet_size = len(packet)

                protocol = "OTHER"
                src_port = 0
                dst_port = 0
                tcp_flags = ""
                tcp_seq = 0
                tcp_window = 0

                if TCP in packet:
                    protocol = "TCP"
                    src_port = int(packet[TCP].sport)
                    dst_port = int(packet[TCP].dport)
                    tcp_flags = str(packet[TCP].flags)
                    tcp_seq = int(packet[TCP].seq)
                    tcp_window = int(packet[TCP].window)
                elif UDP in packet:
                    protocol = "UDP"
                    src_port = int(packet[UDP].sport)
                    dst_port = int(packet[UDP].dport)

                protocol_counts[protocol] += 1

                endpoint_a = (src_ip, src_port)
                endpoint_b = (dst_ip, dst_port)

                if endpoint_a <= endpoint_b:
                    key = (endpoint_a, endpoint_b, protocol)
                else:
                    key = (endpoint_b, endpoint_a, protocol)

                if key not in flows:
                    flows[key] = {
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "src_port": src_port,
                        "dst_port": dst_port,
                        "proto": protocol,
                        "start": timestamp,
                        "end": timestamp,
                        "forward_packets": 0,
                        "backward_packets": 0,
                        "forward_bytes": 0,
                        "backward_bytes": 0,
                        "forward_times": [],
                        "backward_times": [],
                        "forward_ttl": src_ttl,
                        "backward_ttl": dst_ttl,
                        "syn": 0,
                        "fin": 0,
                        "rst": 0,
                        "forward_window": tcp_window,
                        "backward_window": 0,
                        "forward_seq": tcp_seq,
                        "backward_seq": 0,
                    }

                flow = flows[key]
                is_forward = (
                    src_ip == flow["src_ip"]
                    and src_port == flow["src_port"]
                )

                flow["end"] = timestamp

                if is_forward:
                    flow["forward_packets"] += 1
                    flow["forward_bytes"] += packet_size
                    flow["forward_times"].append(timestamp)
                else:
                    flow["backward_packets"] += 1
                    flow["backward_bytes"] += packet_size
                    flow["backward_times"].append(timestamp)

                    if TCP in packet:
                        flow["backward_window"] = tcp_window
                        flow["backward_seq"] = tcp_seq

                if "S" in tcp_flags:
                    flow["syn"] += 1
                if "F" in tcp_flags:
                    flow["fin"] += 1
                if "R" in tcp_flags:
                    flow["rst"] += 1

                line = (
                    f"Packet {packet_count}: {src_ip}:{src_port} → "
                    f"{dst_ip}:{dst_port}, protocol={protocol}, "
                    f"length={packet_size}, TCP_flags={tcp_flags or '-'}"
                )

                if DNS in packet and packet[DNS].qd and DNSQR in packet:
                    dns_query = packet[DNSQR].qname
                    if isinstance(dns_query, bytes):
                        dns_query = dns_query.decode("utf-8", errors="replace")
                    line += f", DNS_query={dns_query}"

                packet_lines.append(line)

    except Exception as exc:
        raise ValueError(f"Could not read this PCAP file: {exc}") from exc

    flow_rows = []

    for flow in flows.values():
        duration = max(flow["end"] - flow["start"], 0.0)
        forward_count = flow["forward_packets"]
        backward_count = flow["backward_packets"]
        forward_bytes = flow["forward_bytes"]
        backward_bytes = flow["backward_bytes"]

        forward_intervals = [
            b - a
            for a, b in zip(
                flow["forward_times"],
                flow["forward_times"][1:],
            )
        ]
        backward_intervals = [
            b - a
            for a, b in zip(
                flow["backward_times"],
                flow["backward_times"][1:],
            )
        ]

        flow_rows.append({
            "src_ip": flow["src_ip"],
            "dst_ip": flow["dst_ip"],
            "src_port": flow["src_port"],
            "dst_port": flow["dst_port"],
            "proto": flow["proto"],
            "dur": duration,
            "spkts": forward_count,
            "dpkts": backward_count,
            "sbytes": forward_bytes,
            "dbytes": backward_bytes,
            "rate": (
                (forward_count + backward_count) / duration
                if duration > 0 else 0
            ),
            "sinpkt": (
                sum(forward_intervals) / len(forward_intervals)
                if forward_intervals else 0
            ),
            "dinpkt": (
                sum(backward_intervals) / len(backward_intervals)
                if backward_intervals else 0
            ),
            "sttl": flow["forward_ttl"],
            "dttl": flow["backward_ttl"],
            "sload": forward_bytes / duration if duration > 0 else 0,
            "dload": backward_bytes / duration if duration > 0 else 0,
            "sloss": 0,
            "dloss": 0,
            "swin": flow["forward_window"],
            "dwin": flow["backward_window"],
            "stcpb": flow["forward_seq"],
            "dtcpb": flow["backward_seq"],
            "tcprtt": 0,
            "synack": 0,
            "ackdat": 0,
            "smean": forward_bytes / forward_count if forward_count else 0,
            "dmean": backward_bytes / backward_count if backward_count else 0,
            "state": "CON" if flow["proto"] == "TCP" else "INT",
            "service": (
                "dns"
                if 53 in (flow["src_port"], flow["dst_port"])
                else "-"
            ),
            "tcp_syn_count": flow["syn"],
            "tcp_fin_count": flow["fin"],
            "tcp_rst_count": flow["rst"],
        })

    summary_lines = [
        f"Packets analyzed: {len(packet_lines)}",
        f"Flows extracted: {len(flow_rows)}",
        "Protocol counts: "
        + ", ".join(
            f"{protocol}={count}"
            for protocol, count in protocol_counts.most_common()
        ),
    ]

    llm_context = "\n".join(
        summary_lines
        + [
            "",
            "Sample packet details:",
            *packet_lines[:MAX_PACKET_LINES_FOR_LLM],
        ]
    )[:MAX_CONTEXT_CHARS_FOR_LLM]

    return {
        "packet_count": len(packet_lines),
        "protocol_counts": protocol_counts,
        "packet_lines": packet_lines,
        "flow_rows": flow_rows,
        "llm_context": llm_context,
    }


# -------------------- ML prediction --------------------

def predict_flows(flow_rows, model, feature_columns, label_map):
    if not flow_rows:
        return pd.DataFrame(), pd.DataFrame()

    raw_features = pd.DataFrame(flow_rows)

    model_features = raw_features.reindex(
        columns=feature_columns,
        fill_value=0,
    )

    model_features = (
        model_features
        .replace([float("inf"), float("-inf")], 0)
        .fillna(0)
    )

    predictions = model.predict(model_features)

    results = raw_features[
        ["src_ip", "src_port", "dst_ip", "dst_port", "proto"]
    ].copy()

    results["ML prediction"] = [
        label_map.get(str(pred), label_map.get(pred, str(pred)))
        for pred in predictions
    ]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(model_features)
        results["Confidence"] = probabilities.max(axis=1).round(3)

    return results, model_features


# -------------------- ML findings page --------------------

if st.session_state.page == "ML findings":
    st.title("ML Findings")
    st.caption("Upload a PCAP to extract traffic flows and run the ML model.")

    uploaded_file = st.file_uploader(
        "Upload a PCAP or PCAPNG file",
        type=["pcap", "pcapng"],
        key="ml_pcap_upload",
    )

    if uploaded_file is not None:
        st.write(f"Selected file: **{uploaded_file.name}**")

        if st.button("Analyze PCAP", type="primary", key="analyze_pcap"):
            with st.spinner("Extracting packet and flow features..."):
                try:
                    result = extract_pcap_context(uploaded_file)
                except ValueError as exc:
                    st.error(str(exc))
                    st.stop()

            col1, col2 = st.columns(2)
            col1.metric("Packets analyzed", result["packet_count"])
            col2.metric("Flows extracted", len(result["flow_rows"]))

            st.subheader("Traffic Summary")
            if result["protocol_counts"]:
                st.write(
                    "Protocol counts: "
                    + ", ".join(
                        f"{protocol}: {count}"
                        for protocol, count in result["protocol_counts"].most_common()
                    )
                )

            with st.expander("Extracted flow features"):
                st.dataframe(
                    pd.DataFrame(result["flow_rows"]),
                    use_container_width=True,
                )

            st.subheader("ML Detection")

            try:
                model, feature_columns, label_map = load_model()
                predictions, model_input = predict_flows(
                    result["flow_rows"],
                    model,
                    feature_columns,
                    label_map,
                )

                if predictions.empty:
                    st.info("No IP-based flows were extracted from this file.")
                else:
                    st.dataframe(predictions, use_container_width=True)

                st.caption(
                    "Experimental output: verify that extracted features and "
                    "preprocessing match the model's training setup."
                )

                with st.expander("Exact feature table passed to model"):
                    st.dataframe(model_input, use_container_width=True)

                st.session_state["pcap_context"] = result["llm_context"]
                st.session_state["ml_context"] = (
                    "ML flow predictions:\n" + predictions.to_string(index=False)
                    if not predictions.empty
                    else "ML prediction was not available."
                )

            except Exception as exc:
                st.error(f"Could not load or run the model: {exc}")
                st.info(
                    "Check Drive sharing is 'Anyone with the link — Viewer' "
                    "and that the saved bundle contains the model and feature columns."
                )
                st.session_state["pcap_context"] = result["llm_context"]
                st.session_state["ml_context"] = "ML prediction was not available."


# -------------------- RAG investigation page --------------------

elif st.session_state.page == "RAG investigation":
    st.title("RAG Investigation")
    st.caption("Ask a question about the analyzed PCAP and review the SOC-style explanation.")

    default_question = (
        "Summarize this traffic and identify anything suspicious."
    )

    question = st.text_area(
        "Investigation question",
        value=default_question,
        height=100,
    )

    if st.button("Run investigation", type="primary", key="run_rag"):
        pcap_context = st.session_state.get("pcap_context", "")
        ml_context = st.session_state.get("ml_context", "")

        if not pcap_context:
            st.warning("Go to ML findings and analyze a PCAP first.")
        else:
            with st.spinner("Generating SOC investigation..."):
                answer = explain_answer(
                    query=question.strip() or default_question,
                    base_answer=pcap_context + "\n\n" + ml_context,
                )

            if answer.startswith("LLM_ERROR:"):
                st.error(answer)
                st.info(
                    "Check GROQ_API_KEY in Streamlit Cloud Secrets and "
                    "confirm the selected Groq model is available."
                )
            else:
                st.markdown(answer)


# -------------------- About page --------------------

elif st.session_state.page == "About":
    st.title("About AI-SOC Threat Analyzer")
    st.write(
        """
        AI-SOC Threat Analyzer is a prototype for reviewing network traffic
        captured in PCAP or PCAPNG files.

        **Main capabilities**
        - Extract packet and flow-level information.
        - Run an ML model against extracted flow features.
        - Generate a SOC-style investigation using the available traffic
          context and ML findings.

        **Important:** ML results are experimental. Validate the feature
        extraction and preprocessing against the model's training data before
        relying on predictions for security decisions.
        """
    )
