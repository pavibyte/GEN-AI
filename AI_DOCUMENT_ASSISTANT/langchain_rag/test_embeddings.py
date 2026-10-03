from langchain_rag.embeddings import get_embeddings


embeddings = get_embeddings()

text = "Employees can work remotely on Fridays."

vector = embeddings.embed_query(text)

print("Embedding dimension:", len(vector))
print("First 5 values:", vector[:5])