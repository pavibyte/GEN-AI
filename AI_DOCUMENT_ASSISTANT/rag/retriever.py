from rag.embeddings import get_embedding
from rag.vector_store import VectorStore


class Retriever:
    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve(
        self,
        query: str,
        n_results: int = 3,
        distance_threshold: float = 0.60
    ):
        query_embedding = get_embedding(query)

        results = self.vector_store.search(
            query_embedding,
            n_results
        )

        documents = results["documents"][0]
        distances = results["distances"][0]

        metadatas = results["metadatas"][0]

        filtered_documents = []

        for document, distance, metadata in zip(
            documents,
            distances,
            metadatas
        ):
            if distance <= distance_threshold:
                filtered_documents.append(
                    {
                        "document": document,
                        "distance": distance,
                        "metadata": metadata
                    }
                )

        return filtered_documents