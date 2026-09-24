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

question = "What is the company's maternity leave policy?"

retriever = Retriever()

results = retriever.retrieve(
    question,
    n_results=3
)
print("\nQuestion:")
print(question)
retrieved_documents = results["documents"][0]
print("\nRetrieval Results:")

for i, document in enumerate(results["documents"][0]):
    distance = results["distances"][0][i]

    print(f"\nResult {i + 1}")
    print(f"Distance: {distance}")
    print(f"Document: {document}")


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