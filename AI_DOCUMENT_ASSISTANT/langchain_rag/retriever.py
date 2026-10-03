from langchain_chroma import Chroma


def create_retriever(
    vector_store: Chroma,
    k: int = 3
):
    return vector_store.as_retriever(
        search_kwargs={
            "k": k
        }
    )