from langchain_core.documents import Document


def create_documents(text: str) -> list[Document]:

    sections = text.split("\n\n")

    documents = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        first_line = section.split("\n")[0]
        section_name = first_line.rstrip(":")

        documents.append(
            Document(
                page_content=section,
                metadata={
                    "source": "company_policy.txt",
                    "section": section_name
                }
            )
        )

    return documents