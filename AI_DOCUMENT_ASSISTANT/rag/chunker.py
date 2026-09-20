def chunk_text(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50
) -> list[str]:

    if overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    separators = ["\n\n", "\n", ". ", " "]

    chunks = []

    def split_text(text_part: str, separator_index: int):

        if len(text_part) <= chunk_size:
            chunks.append(text_part.strip())
            return

        if separator_index >= len(separators):
            for start in range(
                0,
                len(text_part),
                chunk_size - overlap
            ):
                chunk = text_part[start:start + chunk_size]
                chunks.append(chunk.strip())
            return

        separator = separators[separator_index]

        parts = text_part.split(separator)

        current_chunk = ""

        for part in parts:

            candidate = (
                current_chunk + separator + part
                if current_chunk
                else part
            )

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:

                if current_chunk:
                    chunks.append(current_chunk.strip())

                current_chunk = part

        if current_chunk:
            split_text(
                current_chunk,
                separator_index + 1
            )

    split_text(text, 0)

    return [chunk for chunk in chunks if chunk]