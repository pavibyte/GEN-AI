from rag.embeddings import get_embedding
from rag.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        n_results: int = 3
    ):

        query_embedding = get_embedding(query)

        results = self.vector_store.search(
            query_embedding,
            n_results
        )

        return results