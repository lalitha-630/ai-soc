import os

from groq import Groq


MODEL_NAME = "openai/gpt-oss-20b"
MAX_QUERY_CHARS = 1500
MAX_EVIDENCE_CHARS = 5000


def get_api_key():
    """Read the Groq API key from environment variables or Streamlit secrets."""
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            api_key = None

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env file "
            "or Streamlit Cloud Secrets."
        )

    return api_key


def explain_answer(query: str, base_answer: str) -> str:
    """Explain supplied PCAP/ML findings in a concise SOC analyst style."""
    try:
        query = (query or "").strip()[:MAX_QUERY_CHARS]
        evidence = (base_answer or "").strip()[:MAX_EVIDENCE_CHARS]

        if not query:
            query = "Investigate the supplied network traffic findings."

        if not evidence:
            return "No PCAP or ML findings were provided for investigation."

        client = Groq(api_key=get_api_key())

        response = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.3,
            max_tokens=900,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a SOC analyst assisting with network traffic "
                        "triage. Use only the supplied evidence. Do not claim "
                        "an attack is confirmed unless the evidence supports it. "
                        "Distinguish observations from hypotheses. If evidence "
                        "is insufficient, say so. Keep the response under "
                        "400 words and concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Analyst question:\n{query}\n\n"
                        f"PCAP and ML evidence:\n{evidence}\n\n"
                        "Provide:\n"
                        "1. Summary of observed traffic\n"
                        "2. Potential security concern, if supported\n"
                        "3. Risk level (Low, Medium, High, or Inconclusive) "
                        "and the reason\n"
                        "4. Recommended investigation or response steps\n\n"
                        "Treat ML labels as predictions, not confirmed facts. "
                        "Do not infer packet details that are not in the evidence."
                    ),
                },
            ],
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = choice.finish_reason

        if not content.strip():
            if finish_reason == "length":
                return (
                    "The investigation response exceeded the model's output "
                    "limit. Retry with a more focused question or a smaller "
                    "findings PDF."
                )

            return (
                "The model returned an empty response. "
                f"Finish reason: {finish_reason or 'not provided'}."
            )

        if finish_reason == "length":
            content = content.rstrip() + "\n\n(Response may be incomplete.)"

        return content.strip()

    except Exception as exc:
        return f"LLM_ERROR: {type(exc).__name__}: {exc}"
