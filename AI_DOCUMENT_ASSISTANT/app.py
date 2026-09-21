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

chunks = chunk_text(
    document,
    chunk_size=300,
    overlap=50
)

embeddings = get_embeddings(chunks)

vector_store = VectorStore()

vector_store.add_documents(
    chunks,
    embeddings
)


# -------------------------
# 2. RETRIEVAL
# -------------------------

question = "How many vacation days do employees get?"

retriever = Retriever()

results = retriever.retrieve(
    question,
    n_results=3
)

retrieved_documents = results["documents"][0]


# -------------------------
# 3. GENERATION
# -------------------------

context = "\n\n".join(
    retrieved_documents
)

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