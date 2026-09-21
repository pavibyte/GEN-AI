from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load the existing .env file
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env", override=True)


client = genai.Client()

MODEL = "gemini-3.6-flash"


def generate_answer(
    question: str,
    context: str
) -> str:

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using only the information
provided in the context.

If the answer cannot be found in the context,
say:

"I don't have enough information to answer that."

Context:
{context}

Question:
{question}

Answer:
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text