from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import get_embeddings
from rag.vector_store import VectorStore


# 1. Load document
document = load_document(
    "data/company_policy.txt"
)

# 2. Create chunks
chunks = chunk_text(
    document,
    chunk_size=300,
    overlap=50
)

# 3. Create embeddings
embeddings = get_embeddings(chunks)

# 4. Store in ChromaDB
vector_store = VectorStore()

vector_store.add_documents(
    chunks,
    embeddings
)

print(f"Stored {len(chunks)} chunks in ChromaDB.")