import chromadb


class VectorStore:

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_db"
        )

        self.collection = self.client.get_or_create_collection(
            name="company_knowledge"
        )

    def add_documents(
        self,
        chunks: list[str],
        embeddings: list[list[float]]
    ):

        ids = [
            f"policy_chunk_{i}"
            for i in range(len(chunks))
        ]

        metadatas = [
            {
                "source": "company_policy.txt"
            }
            for _ in chunks
        ]

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_embedding: list[float],
        n_results: int = 3
    ):

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )