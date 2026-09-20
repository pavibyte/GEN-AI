from rag.document_loader import load_document
from rag.chunker import chunk_text
from rag.embeddings import get_embeddings


document = load_document("data/company_policy.txt")

chunks = chunk_text(
    document,
    chunk_size=300,
    overlap=50
)

embeddings = get_embeddings(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Total embeddings: {len(embeddings)}")

for i, embedding in enumerate(embeddings, start=1):
    print(f"\nChunk {i}")
    print(f"Vector length: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")