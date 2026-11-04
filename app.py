import io
import os
from collections import Counter
from xml.sax.saxutils import escape

import gdown
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pypdf import PdfReader
from scapy.all import PcapReader, IP, IPv6, TCP, UDP, DNS, DNSQR
from llm import explain_answer

st.set_page_config(page_title="AI-SOC Threat Analyzer", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
:root { --navy:#07172b; --panel:#102744; --panel2:#163554; --blue:#9cc9ff; --blue2:#5da8f5; --line:#315578; --text:#eaf4ff; }
.stApp { background:var(--navy); color:var(--text); font-size:20px; }
[data-testid="stHeader"] { background:rgba(7,23,43,.96); }
[data-testid="stSidebar"] { background:#0b2039; border-right:1px solid var(--line); min-width:290px; width:290px; }
[data-testid="stSidebar"] > div:first-child { padding:1.4rem 1rem; }
[data-testid="stSidebar"] * { color:var(--text); }
h1 { color:var(--blue) !important; font-size:2.7rem !important; line-height:1.15; }
h2 { color:var(--blue) !important; font-size:2.15rem !important; }
h3 { color:var(--blue) !important; font-size:1.65rem !important; }
p, label, [data-testid="stMarkdownContainer"] { color:var(--text); font-size:20px; }
small, [data-testid="stCaptionContainer"] { color:#b9d7f6 !important; font-size:17px !important; }
[data-testid="stSidebar"] .stButton > button {
 width:100%; min-height:62px; text-align:left; font-size:22px; font-weight:700;
 border-radius:12px; padding:13px 16px; background:var(--panel); color:var(--blue);
 border:1px solid var(--line); transition:all .18s ease;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] .stButton > button:focus,
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
 background:var(--blue2); color:var(--navy); border-color:var(--blue);
}
.stButton > button, .stDownloadButton > button {
 min-height:54px; font-size:20px; font-weight:700; background:var(--panel2);
 color:var(--blue); border:1px solid var(--blue2); border-radius:11px; transition:all .18s ease;
}
.stButton > button:hover, .stDownloadButton > button:hover { background:var(--blue); color:var(--navy); border-color:var(--blue); }
[data-testid="stMetric"] { background:linear-gradient(145deg,#102744,#163554); border:1px solid var(--line); padding:22px; border-radius:14px; min-height:132px; }
[data-testid="stMetricLabel"] { color:#b9d7f6 !important; font-size:19px !important; }
[data-testid="stMetricValue"] { color:var(--blue) !important; font-size:42px !important; }
[data-testid="stFileUploader"] section, [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] { background:var(--panel) !important; color:var(--text) !important; border:1px solid var(--line) !important; border-radius:12px; }
[data-testid="stFileUploader"] section * { color:var(--text) !important; font-size:18px !important; }
.stTextInput input, .stTextArea textarea, .stSelectbox [data-baseweb="select"], .stNumberInput input, .stDateInput input { background:var(--panel) !important; color:var(--text) !important; font-size:19px !important; border:1px solid var(--line) !important; border-radius:9px; }
.stTextInput label, .stTextArea label, .stSelectbox label { color:var(--blue) !important; font-size:19px !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:10px; }
[data-testid="stExpander"] { background:var(--panel); border:1px solid var(--line); border-radius:11px; margin-bottom:18px; }
[data-testid="stExpander"] summary { color:var(--blue); font-size:20px; font-weight:700; }
[data-testid="stAlert"] { font-size:19px; }
[data-testid="stPlotlyChart"] { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:10px; }
hr { border-color:var(--line); }
</style>
""", unsafe_allow_html=True)

MAX_PACKETS = 10000
MODEL_PATH = "/tmp/model_bundle.joblib"
DRIVE_ID = "1QSBGKdxnXwX1gzG1n_EO6rMAznARfDpy"
PAGES = ["Dashboard", "ML Findings", "AI Analyst", "Statistics", "About"]

defaults = {
    "page": "Dashboard", "pcap_context": "", "ml_context": "",
    "flow_rows": [], "packet_lines": [], "protocol_counts": {},
    "packet_count": 0, "pcap_filename": "", "predictions": None,
    "model_input": None, "pcap_analyzed": False, "ai_answer": ""
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

with st.sidebar:
    st.divider()
    nav_labels = {
        "Dashboard": "📊  Dashboard",
        "ML Findings": "🧠  ML Findings",
        "AI Analyst": "🤖  AI Analyst",
        "Statistics": "📈  Statistics",
        "About": "ℹ️  About",
    }
    for page in PAGES:
        if st.button(nav_labels[page], key=f"nav_{page}", type="primary" if st.session_state.page == page else "secondary"):
            st.session_state.page = page
            st.rerun()


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={DRIVE_ID}"
        path = gdown.download(url, output=MODEL_PATH, quiet=False)
        if not path or not os.path.exists(MODEL_PATH):
            raise RuntimeError("Model download failed. Check Drive link sharing.")
    loaded = joblib.load(MODEL_PATH)
    if isinstance(loaded, dict):
        model = loaded.get("pipeline") or loaded.get("model")
        columns = loaded.get("feature_columns")
        label_map = loaded.get("label_map", {})
    else:
        model = loaded
        columns = getattr(model, "feature_names_in_", None)
        label_map = {}
    if model is None:
        raise ValueError("Model bundle must contain 'pipeline' or 'model'.")
    if columns is None:
        columns = getattr(model, "feature_names_in_", None)
    if columns is None:
        raise ValueError("Model feature columns were not found in the bundle.")
    return model, list(columns), label_map


def packet_ip_pair(packet):
    if IP in packet:
        return packet[IP].src, packet[IP].dst, packet[IP].ttl, packet[IP].ttl
    if IPv6 in packet:
        return packet[IPv6].src, packet[IPv6].dst, 0, 0
    return None


def extract_pcap_context(uploaded_file, max_packets=MAX_PACKETS):
    flows, packet_lines = {}, []
    protocols = Counter()
    with PcapReader(io.BytesIO(uploaded_file.getvalue())) as reader:
        try:
            for index, packet in enumerate(reader, start=1):
                if index > max_packets:
                    break
                pair = packet_ip_pair(packet)
                if not pair:
                    continue
                src, dst, src_ttl, dst_ttl = pair
                proto, sport, dport, flags, seq, window = "OTHER", 0, 0, "", 0, 0
                if TCP in packet:
                    proto = "TCP"
                    sport, dport = int(packet[TCP].sport), int(packet[TCP].dport)
                    flags, seq, window = str(packet[TCP].flags), int(packet[TCP].seq), int(packet[TCP].window)
                elif UDP in packet:
                    proto = "UDP"
                    sport, dport = int(packet[UDP].sport), int(packet[UDP].dport)
                protocols[proto] += 1
                a, b = (src, sport), (dst, dport)
                key = (a, b, proto) if a <= b else (b, a, proto)
                if key not in flows:
                    flows[key] = dict(src_ip=src, dst_ip=dst, src_port=sport, dst_port=dport,
                        proto=proto, start=float(packet.time), end=float(packet.time),
                        forward_packets=0, backward_packets=0, forward_bytes=0, backward_bytes=0,
                        forward_times=[], backward_times=[], forward_ttl=src_ttl, backward_ttl=dst_ttl,
                        syn=0, fin=0, rst=0, forward_window=window, backward_window=0,
                        forward_seq=seq, backward_seq=0)
                f = flows[key]
                forward = src == f["src_ip"] and sport == f["src_port"]
                f["end"] = float(packet.time)
                direction = "forward" if forward else "backward"
                f[f"{direction}_packets"] += 1
                f[f"{direction}_bytes"] += len(packet)
                f[f"{direction}_times"].append(float(packet.time))
                if not forward and TCP in packet:
                    f["backward_window"], f["backward_seq"] = window, seq
                f["syn"] += int("S" in flags)
                f["fin"] += int("F" in flags)
                f["rst"] += int("R" in flags)
                line = f"Packet {index}: {src}:{sport} -> {dst}:{dport}, protocol={proto}, length={len(packet)}, TCP_flags={flags or '-'}"
                if DNS in packet and packet[DNS].qd and DNSQR in packet:
                    qname = packet[DNSQR].qname
                    if isinstance(qname, bytes):
                        qname = qname.decode("utf-8", errors="replace")
                    line += f", DNS_query={qname}"
                packet_lines.append(line)
        except Exception as exc:
            raise ValueError(f"Could not read this PCAP file: {exc}") from exc

    rows = []
    for f in flows.values():
        duration = max(f["end"] - f["start"], 0)
        fc, bc = f["forward_packets"], f["backward_packets"]
        fb, bb = f["forward_bytes"], f["backward_bytes"]
        fi = [y-x for x, y in zip(f["forward_times"], f["forward_times"][1:])]
        bi = [y-x for x, y in zip(f["backward_times"], f["backward_times"][1:])]
        rows.append({
            "src_ip": f["src_ip"], "dst_ip": f["dst_ip"], "src_port": f["src_port"],
            "dst_port": f["dst_port"], "proto": f["proto"], "dur": duration,
            "spkts": fc, "dpkts": bc, "sbytes": fb, "dbytes": bb,
            "rate": (fc+bc)/duration if duration else 0,
            "sinpkt": sum(fi)/len(fi) if fi else 0, "dinpkt": sum(bi)/len(bi) if bi else 0,
            "sttl": f["forward_ttl"], "dttl": f["backward_ttl"],
            "sload": fb/duration if duration else 0, "dload": bb/duration if duration else 0,
            "sloss": 0, "dloss": 0, "swin": f["forward_window"], "dwin": f["backward_window"],
            "stcpb": f["forward_seq"], "dtcpb": f["backward_seq"], "tcprtt": 0,
            "synack": 0, "ackdat": 0, "smean": fb/fc if fc else 0, "dmean": bb/bc if bc else 0,
            "state": "CON" if f["proto"] == "TCP" else "INT",
            "service": "dns" if 53 in (f["src_port"], f["dst_port"]) else "-",
            "tcp_syn_count": f["syn"], "tcp_fin_count": f["fin"], "tcp_rst_count": f["rst"]
        })
    context = "\n".join([
        f"Packets analyzed: {len(packet_lines)}", f"Flows extracted: {len(rows)}",
        "Protocol counts: " + ", ".join(f"{p}={n}" for p, n in protocols.most_common()),
        "", "Sample packet details:", *packet_lines[:30]
    ])[:5000]
    return dict(packet_count=len(packet_lines), protocol_counts=protocols,
                packet_lines=packet_lines, flow_rows=rows, llm_context=context)


def predict_flows(rows, model, columns, label_map):
    if not rows:
        return pd.DataFrame(), pd.DataFrame()
    raw = pd.DataFrame(rows)
    features = raw.reindex(columns=columns, fill_value=0).replace([float("inf"), float("-inf")], 0).fillna(0)
    preds = model.predict(features)
    result = raw[["src_ip", "src_port", "dst_ip", "dst_port", "proto"]].copy()
    result["ML prediction"] = [label_map.get(str(p), label_map.get(p, str(p))) for p in preds]
    if hasattr(model, "predict_proba"):
        result["Confidence"] = model.predict_proba(features).max(axis=1).round(3)
    return result, features


def severity(label):
    text = str(label).lower()
    if any(x in text for x in ("malicious", "attack", "intrusion", "botnet", "critical")):
        return "High"
    if any(x in text for x in ("suspicious", "anomaly", "scan")):
        return "Medium"
    if any(x in text for x in ("normal", "benign")):
        return "Low"
    return "Review"


def make_findings_pdf(summary, findings):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28)
    styles = getSampleStyleSheet()
    story = [Paragraph("AI-SOC Threat Analyzer — ML Findings", styles["Title"]),
             Spacer(1, 10), Paragraph("PCAP Summary", styles["Heading2"]),
             Paragraph(escape(summary).replace("\n", "<br/>"), styles["BodyText"]),
             Spacer(1, 12), Paragraph("ML Predictions", styles["Heading2"])]
    if findings is None or findings.empty:
        story.append(Paragraph("No ML predictions available.", styles["BodyText"]))
    else:
        data = [list(findings.columns)] + findings.fillna("").astype(str).values.tolist()
        table = Table(data, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1E3A5F")),
            ("TEXTCOLOR", (0,0), (-1,0), colors.white),
            ("GRID", (0,0), (-1,-1), 0.35, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 7), ("VALIGN", (0,0), (-1,-1), "TOP")
        ]))
        story.append(table)
    doc.build(story)
    return buffer.getvalue()


# Dashboard
header_path = os.path.join(os.path.dirname(__file__), "soc_header.png")
if os.path.exists(header_path):
    st.image(header_path, use_container_width=True)

if st.session_state.page == "Dashboard":
    st.markdown(
        "<h1 style='font-size:36px!important; margin:18px 0 12px;'>UPLOAD PCAP FILE</h1>",
        unsafe_allow_html=True,
    )
    capture = st.file_uploader(
        "Choose a PCAP or PCAPNG file",
        type=["pcap", "pcapng"],
        key="dashboard_pcap",
    )
    if capture:
        st.caption(f"Selected file: {capture.name}")

    if st.button("ANALYZE PCAP", type="primary", key="analyze", use_container_width=True):
        if capture is None:
            st.warning("Upload a PCAP/PCAPNG file before analyzing.")
        else:
            try:
                with st.spinner("Analyzing packets and extracting flows..."):
                    result = extract_pcap_context(capture)
                    model, cols, labels = load_model()
                    findings, model_input = predict_flows(result["flow_rows"], model, cols, labels)
                st.session_state.pcap_context = result["llm_context"]
                st.session_state.ml_context = "ML flow predictions:\\n" + findings.to_string(index=False)
                st.session_state.flow_rows = result["flow_rows"]
                st.session_state.packet_lines = result["packet_lines"]
                st.session_state.protocol_counts = dict(result["protocol_counts"])
                st.session_state.packet_count = result["packet_count"]
                st.session_state.pcap_filename = capture.name
                st.session_state.predictions = findings
                st.session_state.model_input = model_input
                st.session_state.pcap_analyzed = True
                st.session_state.ai_answer = ""
                st.success("Analysis complete.")
            except Exception as exc:
                st.error(f"PCAP analysis failed: {exc}")

    if st.session_state.pcap_analyzed:
        st.markdown(
            f"<h2 style='font-size:28px!important;'>Analysis complete — "
            f"{st.session_state.packet_count} packets · {len(st.session_state.flow_rows)} flows</h2>",
            unsafe_allow_html=True,
        )
        with st.expander("ACTUAL PACKETS", expanded=False):
            packet_search = st.text_input("Search packet records", key="dashboard_packet_search")
            packet_rows = st.session_state.packet_lines
            if packet_search:
                packet_rows = [p for p in packet_rows if packet_search.lower() in p.lower()]
            st.dataframe(
                pd.DataFrame({"Packet details": packet_rows}),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("EXTRACTED FLOWS", expanded=False):
            flow_frame = pd.DataFrame(st.session_state.flow_rows)
            flow_search = st.text_input("Search extracted flows", key="dashboard_flow_search")
            if flow_search and not flow_frame.empty:
                matches = flow_frame.astype(str).apply(
                    lambda col: col.str.contains(flow_search, case=False, na=False)
                ).any(axis=1)
                flow_frame = flow_frame[matches]
            st.dataframe(flow_frame, use_container_width=True, hide_index=True)
# ML Findings
elif st.session_state.page == "ML Findings":
    st.markdown(
        "<h1 style='font-size:36px!important; line-height:1.2; margin:0 0 18px;'>ML Findings</h1>",
        unsafe_allow_html=True,
    )
    if not st.session_state.pcap_analyzed:
        st.info("Analyze a PCAP from Dashboard first.")
    else:
        st.caption(f"PCAP: {st.session_state.pcap_filename}")
        findings = st.session_state.predictions
        if findings is not None and not findings.empty:
            shown = findings.copy()
            shown["Severity"] = shown["ML prediction"].apply(severity)
            st.dataframe(shown, use_container_width=True)
            st.download_button("Download ML Findings as PDF",
                data=make_findings_pdf(st.session_state.pcap_context, shown),
                file_name="ai_soc_ml_findings.pdf", mime="application/pdf", type="primary")
        else:
            st.warning("No flow predictions available.")


# AI Analyst
elif st.session_state.page == "AI Analyst":
    st.title("AI Analyst")
    st.caption("Upload the ML findings PDF and ask for an investigation.")
    report = st.file_uploader("Upload ML findings PDF", type=["pdf"], key="findings_pdf")
    question = st.text_area("Investigation question",
        value="Summarize the findings and identify anything that needs investigation.", height=100)
    if st.button("Investigate", type="primary", key="investigate"):
        if report is None:
            st.warning("Upload the ML findings PDF first.")
        else:
            try:
                reader = PdfReader(io.BytesIO(report.getvalue()))
                text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
                if not text:
                    st.warning("No selectable text found. Upload a text-based PDF.")
                else:
                    context = "Uploaded ML findings PDF:\n" + text[:12000]
                    if st.session_state.pcap_context:
                        context += "\n\nPCAP context:\n" + st.session_state.pcap_context
                    if st.session_state.ml_context:
                        context += "\n\nML findings:\n" + st.session_state.ml_context
                    with st.spinner("Investigating..."):
                        st.session_state.ai_answer = explain_answer(
                            query=question.strip() or "Summarize the findings.",
                            base_answer=context)
            except Exception as exc:
                st.error(f"Could not process PDF: {exc}")
    if st.session_state.ai_answer:
        if st.session_state.ai_answer.startswith("LLM_ERROR:"):
            st.error(st.session_state.ai_answer)
        else:
            st.subheader("Investigation result")
            st.markdown(st.session_state.ai_answer)


# Statistics
elif st.session_state.page == "Statistics":
    st.title("Traffic & Threat Statistics")
    if not st.session_state.pcap_analyzed:
        st.info("Analyze a PCAP from Dashboard first.")
    else:
        flows = pd.DataFrame(st.session_state.flow_rows)
        findings = st.session_state.predictions
        findings = findings.copy() if findings is not None else pd.DataFrame()
        navy, blue, pale, line = "#07172b", "#5da8f5", "#9cc9ff", "#315578"
        chart_config = {"displayModeBar": True, "displaylogo": False, "responsive": True}

        # Attach model labels and confidence to extracted flows in their original row order.
        if not flows.empty and not findings.empty:
            flows = flows.reset_index(drop=True)
            findings = findings.reset_index(drop=True)
            flows["prediction"] = findings["ML prediction"].astype(str)
            flows["confidence"] = pd.to_numeric(findings.get("Confidence", 0), errors="coerce")
            flows["confidence"] = flows["confidence"].fillna(0).clip(0, 1)
            suspicious_mask = flows["prediction"].str.lower().str.contains(
                "attack|malicious|intrusion|botnet|suspicious|anomaly|scan", regex=True
            )
            # Confidence is the model's confidence in its predicted class. Convert it to
            # a display score where suspicious predictions are high and other predictions low.
            flows["threat_score"] = flows["confidence"].where(suspicious_mask, 1 - flows["confidence"])
        else:
            suspicious_mask = pd.Series(False, index=flows.index)
            flows["threat_score"] = 0.0

        threshold = 0.5
        suspicious_count = int(suspicious_mask.sum()) if len(flows) else 0
        safe_count = max(len(flows) - suspicious_count, 0)

        st.caption(f"Capture: {st.session_state.pcap_filename} · {st.session_state.packet_count:,} packets · {len(flows):,} flows")

        # Top row: per-flow risk and safe/suspicious share.
        left, right = st.columns(2)
        with left:
            st.subheader("Per-flow threat score")
            if not flows.empty:
                scores = flows["threat_score"].astype(float).tolist()
                fig = go.Figure()
                for i, score in enumerate(scores, start=1):
                    fig.add_trace(go.Bar(
                        x=[i], y=[score], name="Flow score",
                        marker_color=blue if score < threshold else pale,
                        showlegend=False,
                        hovertemplate=f"Flow {i}<br>Threat score: %{{y:.3f}}<extra></extra>"
                    ))
                fig.add_hline(y=threshold, line_dash="dash", line_color=pale,
                              annotation_text="Risk threshold", annotation_position="top left")
                fig.update_layout(
                    height=350, paper_bgcolor=navy, plot_bgcolor=navy,
                    font=dict(color=pale, size=15),
                    xaxis_title="Flow", yaxis_title="Threat score (0–1)",
                    yaxis=dict(range=[0, 1], gridcolor=line),
                    xaxis=dict(gridcolor=line), margin=dict(l=45, r=20, t=25, b=45),
                    hovermode="closest"
                )
                st.plotly_chart(fig, use_container_width=True, config=chart_config)
            else:
                st.info("No flow scores available.")
        with right:
            st.subheader("Traffic breakdown")
            fig = go.Figure(go.Pie(
                labels=["Safe / other", "Suspicious"], values=[safe_count, suspicious_count],
                hole=.58, marker=dict(colors=[blue, pale], line=dict(color=navy, width=2)),
                textinfo="percent", hovertemplate="%{label}: %{value} flows (%{percent})<extra></extra>"
            ))
            fig.update_layout(height=350, paper_bgcolor=navy, font=dict(color=pale, size=15),
                              legend=dict(orientation="h", y=-.08), margin=dict(l=10,r=10,t=15,b=35),
                              annotations=[dict(text=f"{len(flows)}<br>Flows", x=.5, y=.5, showarrow=False,
                                                 font=dict(size=24, color=pale))])
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

        # Second row: protocol and service distribution.
        left, right = st.columns(2)
        with left:
            st.subheader("Protocol distribution")
            proto_counts = flows["proto"].astype(str).str.upper().value_counts() if not flows.empty else pd.Series(dtype=int)
            fig = go.Figure(go.Bar(x=proto_counts.index.tolist(), y=proto_counts.values.tolist(),
                                   marker_color=blue, hovertemplate="%{x}: %{y} flows<extra></extra>"))
            fig.update_layout(height=320, paper_bgcolor=navy, plot_bgcolor=navy,
                              font=dict(color=pale, size=15), xaxis_title="Protocol", yaxis_title="Flows",
                              xaxis=dict(gridcolor=line), yaxis=dict(gridcolor=line),
                              margin=dict(l=45,r=15,t=15,b=45))
            st.plotly_chart(fig, use_container_width=True, config=chart_config)
        with right:
            st.subheader("Top services contacted")
            if not flows.empty:
                service_series = flows["service"].fillna("-").astype(str).replace({"": "-"})
                service_counts = service_series.value_counts().head(8)
            else:
                service_counts = pd.Series(dtype=int)
            fig = go.Figure(go.Bar(x=service_counts.values.tolist(), y=service_counts.index.tolist(),
                                   orientation="h", marker_color=pale,
                                   hovertemplate="%{y}: %{x} flows<extra></extra>"))
            fig.update_layout(height=320, paper_bgcolor=navy, plot_bgcolor=navy,
                              font=dict(color=pale, size=15), xaxis_title="Flows", yaxis_title="Service",
                              xaxis=dict(gridcolor=line), yaxis=dict(gridcolor=line, autorange="reversed"),
                              margin=dict(l=55,r=15,t=15,b=45))
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

        # Lower charts: score distribution and sent/received volume per flow.
        st.subheader("Threat score distribution — how risky are the connections?")
        st.caption("Each bar counts flows within a threat-score range. The dashed line marks the configured risk boundary.")
        if not flows.empty:
            fig = go.Figure(go.Histogram(
                x=flows["threat_score"], xbins=dict(start=0, end=1, size=.1),
                marker_color=blue, hovertemplate="Score range: %{x}<br>Flows: %{y}<extra></extra>"
            ))
            fig.add_vline(x=threshold, line_dash="dash", line_color=pale,
                          annotation_text="Risk boundary", annotation_position="top")
            fig.update_layout(height=330, paper_bgcolor=navy, plot_bgcolor=navy,
                              font=dict(color=pale, size=15), xaxis_title="Threat score (0 = lower risk, 1 = higher risk)",
                              yaxis_title="Number of flows", xaxis=dict(range=[0,1], gridcolor=line),
                              yaxis=dict(gridcolor=line), margin=dict(l=55,r=20,t=25,b=55))
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

        st.subheader("Data volume per connection")
        st.caption("Shows bytes sent and received per flow; a large transfer is a triage clue, not proof of data theft.")
        if not flows.empty:
            volume = flows.head(30).copy()
            volume["flow_number"] = range(1, len(volume) + 1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=volume["flow_number"], y=volume["sbytes"], mode="lines+markers",
                                     name="Sent (bytes)", line=dict(color=blue, width=3),
                                     hovertemplate="Flow %{x}<br>Sent: %{y:,} bytes<extra></extra>"))
            fig.add_trace(go.Scatter(x=volume["flow_number"], y=volume["dbytes"], mode="lines+markers",
                                     name="Received (bytes)", line=dict(color=pale, width=3),
                                     hovertemplate="Flow %{x}<br>Received: %{y:,} bytes<extra></extra>"))
            fig.update_layout(height=360, paper_bgcolor=navy, plot_bgcolor=navy,
                              font=dict(color=pale, size=15), xaxis_title="Flow number", yaxis_title="Bytes",
                              xaxis=dict(gridcolor=line), yaxis=dict(gridcolor=line),
                              legend=dict(orientation="h", y=1.12), margin=dict(l=55,r=20,t=30,b=50))
            st.plotly_chart(fig, use_container_width=True, config=chart_config)

        st.caption("Threat score is derived from the model's predicted-class confidence and label. It is a visualization aid, not a calibrated probability of compromise.")


# View PCAP File
elif st.session_state.page == "View PCAP File":
    st.title("View PCAP File")
    if not st.session_state.pcap_analyzed:
        st.info("Analyze a PCAP from Dashboard first.")
    else:
        st.write(f"File: **{st.session_state.pcap_filename}**")
        st.metric("Packets analyzed", st.session_state.packet_count)
        search = st.text_input("Search packet details")
        packets = st.session_state.packet_lines
        if search:
            packets = [p for p in packets if search.lower() in p.lower()]
        st.write(f"Showing {len(packets)} packet records")
        st.dataframe(pd.DataFrame({"Packet details": packets}), use_container_width=True)


# Flows
elif st.session_state.page == "Flows":
    st.title("Flows")
    if not st.session_state.pcap_analyzed:
        st.info("Analyze a PCAP from Dashboard first.")
    else:
        flows = pd.DataFrame(st.session_state.flow_rows)
        search = st.text_input("Search flows")
        if search and not flows.empty:
            mask = flows.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
            flows = flows[mask]
        st.write(f"Showing {len(flows)} flows")
        st.dataframe(flows, use_container_width=True)


# About
elif st.session_state.page == "About":
    st.title("About AI-SOC Threat Analyzer")
    st.subheader("Main features")
    st.markdown("""
    - Analyze PCAP/PCAPNG traffic and extract network flows.
    - Classify flows using the saved ML model.
    - Download ML findings as a PDF and upload it to AI Analyst.
    - Review PCAP details, flows, and traffic statistics.
    """)
    st.subheader("Dataset")
    st.write("Add your exact dataset name, source, row count, feature count, label column, and classes here.")
    st.subheader("Machine learning model")
    st.write("The saved model bundle is loaded from Google Drive. Extracted feature names and preprocessing must match training.")
    st.subheader("AI Analyst workflow")
    st.markdown("""
    1. Analyze a PCAP and generate flow-level predictions.
    2. Download the ML findings PDF.
    3. Upload the PDF to AI Analyst and ask a question.
    4. The PDF text and available PCAP context are sent to Groq for a SOC-style explanation.

    This implementation passes extracted PDF text as context; it does not implement embedding-based PDF retrieval.
    """)
