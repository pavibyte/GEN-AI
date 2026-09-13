from google import genai
from dotenv import load_dotenv

load_dotenv(override=True)

client = genai.Client()

texts = [
    "I love this phone.",
    "This smartphone is excellent.",
    "The weather is rainy today."
]

for text in texts:

    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embedding = response.embeddings[0].values

    print("\nText:", text)
    print("Vector length:", len(embedding))
    print("First 10 values:", embedding[:10])