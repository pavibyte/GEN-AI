from langchain_rag.llm import get_llm


llm = get_llm()

response = llm.invoke(
    "Explain what an embedding is in one sentence."
)

print("Response type:")
print(type(response))

print("\nResponse:")
print(response)

print("\nContent type:")
print(type(response.content))

print("\nContent:")
print(response.content)