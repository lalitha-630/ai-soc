import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)


def explain_answer(query: str, base_answer: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # ✅ correct model
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