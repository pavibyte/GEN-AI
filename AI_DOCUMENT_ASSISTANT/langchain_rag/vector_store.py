from langchain_chroma import Chroma

from langchain_rag.embeddings import get_embeddings


def create_vector_store(documents):

    embeddings = get_embeddings()

    vector_store = Chroma(
        collection_name="company_knowledge_langchain",
        embedding_function=embeddings,
        persist_directory="./langchain_chroma_db"
    )

    ids = [
        f"policy_chunk_{i}"
        for i in range(len(documents))
    ]

    vector_store.add_documents(
        documents=documents,
        ids=ids
    )

    return vector_store