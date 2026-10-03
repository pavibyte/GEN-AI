from rag.document_loader import load_document
from langchain_rag.documents import create_documents
from langchain_rag.splitter import split_documents
from langchain_rag.vector_store import create_vector_store


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

results = vector_store.similarity_search(
    "Can employees work remotely?",
    k=3
)

print("\nRetrieved Documents:")

for document in results:
    print("\n---")
    print("Content:")
    print(document.page_content)
    print("Metadata:")
    print(document.metadata)