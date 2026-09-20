from pathlib import Path

from dotenv import load_dotenv
from google import genai


# Load the .env file from the parent GEN AI folder
project_root = Path(__file__).resolve().parents[2]
load_dotenv(project_root / ".env", override=True)


client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"


def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    return response.embeddings[0].values


def get_embeddings(texts: list[str]) -> list[list[float]]:
    return [get_embedding(text) for text in texts]