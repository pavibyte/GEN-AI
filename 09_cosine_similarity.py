from google import genai
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)

client = genai.Client()

texts = [
    "How do I reset my password?",
    "I forgot my account password.",
    "The weather is beautiful today."
]

embeddings = []

for text in texts:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    embeddings.append(response.embeddings[0].values)


def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


similarity_1_2 = cosine_similarity(
    embeddings[0],
    embeddings[1]
)

similarity_1_3 = cosine_similarity(
    embeddings[0],
    embeddings[2]
)

print("Similarity between:")
print(f'"{texts[0]}"')
print(f'"{texts[1]}"')
print("→", similarity_1_2)

print("\nSimilarity between:")
print(f'"{texts[0]}"')
print(f'"{texts[2]}"')
print("→", similarity_1_3)