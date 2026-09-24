def chunk_text(text: str) -> list[str]:
    sections = text.split("\n\n")

    chunks = []

    for section in sections:
        section = section.strip()

        if section:
            chunks.append(section)

    return chunks