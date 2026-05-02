import os
from groq import Groq

# Try loading .env locally (won’t break in cloud if not present)
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# -------------------------------
# Load API Key (LOCAL + CLOUD)
# -------------------------------
api_key = os.getenv("GROQ_API_KEY")

# If not found, try Streamlit secrets
if not api_key:
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except:
        pass

if not api_key:
    raise ValueError("GROQ_API_KEY not found. Set it in .env (local) or Streamlit Secrets (cloud).")

# -------------------------------
# Initialize client
# -------------------------------
client = Groq(api_key=api_key)


# -------------------------------
# LLM Function
# -------------------------------
def explain_answer(query: str, base_answer: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.4,
            max_tokens=300,
            messages=[
                {
                    "role": "system",
                    "content": "You are a SOC analyst. Explain alerts with risk and action."
                },
                {
                    "role": "user",
                    "content": f"""
Query:
{query}

Detection Result:
{base_answer}

Explain clearly:
1. Meaning
2. Attack type
3. Risk level
4. Action
"""
                }
            ],
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"{base_answer}\n\n[GROQ ERROR: {str(e)}]"