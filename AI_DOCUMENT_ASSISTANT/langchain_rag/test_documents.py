from rag.document_loader import load_document
from langchain_rag.documents import create_documents
from langchain_rag.splitter import split_documents


document_text = load_document(
    "data/company_policy.txt"
)

documents = create_documents(document_text)

split_documents_result = split_documents(
    documents
)

print("\nLangChain Documents:")

for document in split_documents_result:
    print("\n---")
    print("Content:")
    print(document.page_content)
    print("Metadata:")
    print(document.metadata)