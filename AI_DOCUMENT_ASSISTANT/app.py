from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import get_embeddings
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import generate_answer


# -------------------------
# 1. INDEXING
# -------------------------

document = load_document(
    "data/company_policy.txt"
)

chunks = chunk_text(document)

embeddings = get_embeddings(chunks)

vector_store = VectorStore()

vector_store.add_documents(
    chunks,
    embeddings
)


# -------------------------
# 2. RETRIEVAL
# -------------------------

question = "What benefits do employees receive from the company?"

retriever = Retriever()

retrieved_documents = retriever.retrieve(
    question,
    n_results=5,
    distance_threshold=0.75
)

print("\nRetrieved Documents:")

for result in retrieved_documents:
    print("\n---")
    print("Document:")
    print(result["document"])
    print("Distance:", result["distance"])
    print("Metadata:", result["metadata"])

print("\nQuestion:")
print(question)

if not retrieved_documents:
    answer = "I don't have enough information to answer that."

else:
    context = "\n\n".join(
    result["document"]
    for result in retrieved_documents
)

    answer = generate_answer(
        question,
        context
    )

source = retrieved_documents[0]["metadata"]["source"]
section = retrieved_documents[0]["metadata"]["section"]

print("\nFinal Answer:")
print(answer)

print(f"\nSource: {source}")
print(f"Section: {section}")

"""
# -------------------------
# 3. GENERATION
# -------------------------

context = "\n\n".join(retrieved_documents)

answer = generate_answer(
    question,
    context
)


# -------------------------
# 4. OUTPUT
# -------------------------

print("\nQuestion:")
print(question)

print("\nRetrieved Context:")
print(context)

print("\nFinal Answer:")
print(answer)
"""