from google import genai
from dotenv import load_dotenv
import chromadb

load_dotenv(override=True)

client = genai.Client()


# 1. Knowledge base

documents = [
    "Employees receive 20 days of annual leave every year.",
    "Employees can work remotely on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Employees receive a performance bonus at the end of the financial year.",
    "The office cafeteria is open from 8 AM to 6 PM."
]

# 2. Create vector database

chroma_client = chromadb.Client()

collection = chroma_client.create_collection(
    name="company_knowledge"
)

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

# 3. User question

question = "What is the company's maternity leave policy?"

# 4. Retrieve relevant documents

results = collection.query(
    query_texts=[question],
    n_results=2
)

retrieved_documents = results["documents"][0]

print("Retrieved documents:")

for document in retrieved_documents:
    print("-", document)

# 5. Augumentation

context = "\n".join(retrieved_documents)

prompt = f"""
Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer cannot be found in the context, say:
"I don't have enough information to answer that."
"""

# 6. Answer generation

response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("\nFinal answer:")
print(response.text)