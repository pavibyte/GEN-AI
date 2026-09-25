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

question = "Can employees work remotely?"

retriever = Retriever()

retrieved_documents = retriever.retrieve(
    question,
    n_results=3,
    distance_threshold=0.60
)

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

print("\nFinal Answer:")
print(answer)


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