import chromadb

client = chromadb.Client()

collection = client.create_collection(
    name="company_documents"
)

documents = [
    "Employees receive 20 days of annual leave every year.",
    "Employees can work remotely on Fridays.",
    "The company provides health insurance to all full-time employees.",
    "Employees receive a performance bonus at the end of the financial year.",
    "The office cafeteria is open from 8 AM to 6 PM."
]

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4", "doc5"]
)

query = "How many vacation days do employees get?"

results = collection.query(
    query_texts=[query],
    n_results=2
)

print(results)

print("\nRelevant documents:")

for document in results["documents"][0]:
    print("-", document)