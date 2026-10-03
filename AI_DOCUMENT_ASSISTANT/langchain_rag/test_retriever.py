from rag.document_loader import load_document
from langchain_rag.documents import create_documents
from langchain_rag.splitter import split_documents
from langchain_rag.vector_store import create_vector_store
from langchain_rag.retriever import create_retriever


document_text = load_document(
    "data/company_policy.txt"
)

documents = create_documents(document_text)

split_documents_result = split_documents(
    documents
)

vector_store = create_vector_store(
    split_documents_result
)

retriever = create_retriever(
    vector_store,
    k=3
)

results = retriever.invoke(
    "Can employees work remotely?"
)

print("\nRetrieved Documents:")

for document in results:
    print("\n---")
    print("Content:")
    print(document.page_content)
    print("Metadata:")
    print(document.metadata)