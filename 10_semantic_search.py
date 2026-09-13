from google import genai
from dotenv import load_dotenv
import numpy as np

load_dotenv(override=True)

client = genai.Client()

documents = [
    "Employees receive 20 days of annual leave every year.",
    "Employees can work remotely on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Employees receive a performance bonus at the end of the financial year.",
    "The office cafeteria is open from 8 AM to 6 PM."
]

query = "How many vacation days do employees get?"


def get_embedding(text):
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )

    return response.embeddings[0].values

# 3. Embeddings for our documents
document_embeddings = []

for document in documents:
    embedding = get_embedding(document)
    document_embeddings.append(embedding)

# 4. user query embedding
query_embedding = get_embedding(query)

# 5. Calculate similarity between query embedding and document embedding
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )

results = []

for document, embedding in zip(documents, document_embeddings):

    score = cosine_similarity(
        query_embedding,
        embedding
    )

    results.append((document, score))

# 6. Ranking the results
results.sort(key=lambda x: x[1], reverse=True)

print("\nSearch results:\n")

for document, score in results:
    print(f"Score: {score:.4f}")
    print(document)
    print()