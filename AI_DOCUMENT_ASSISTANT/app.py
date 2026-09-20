from rag.document_loader import load_document
from rag.chunker import chunk_text


document = load_document("data/company_policy.txt")

chunks = chunk_text(
    document,
    chunk_size=300,
    overlap=50
)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- Chunk {i} ---")
    print(chunk)