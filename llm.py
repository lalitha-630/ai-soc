import os

from groq import Groq


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
    """Explain the supplied PCAP findings in a SOC analyst style."""
    try:
        client = Groq(api_key=get_api_key())

        response = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            temperature=0.3,
            max_tokens=700,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a SOC analyst assisting with network traffic "
                        "triage. Use only the supplied packet evidence. "
                        "Do not claim an attack is confirmed unless the evidence "
                        "supports it. Distinguish observations from hypotheses. "
                        "If evidence is insufficient, say so."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Analyst question:
{query}

PCAP evidence:
{base_answer}

Provide:
1. Summary of observed traffic
2. Potential security concern or attack type, if supported
3. Risk level (Low, Medium, High, or Inconclusive) and reason
4. Recommended investigation or response steps

Keep the answer clear and concise.
""",
                },
            ],
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(
                f"Groq returned empty content. Finish reason: {finish_reason}"
            )

        return content.strip()

    except Exception as exc:
        return f"LLM_ERROR: {type(exc).__name__}: {exc}"
