def count_words(text):
    return len(text.split())

def recursive_chunk(text, max_words=256, overlap_words=60):
    split_by = ["\n\n", "\n", ".", " "]

    def split_text(text, level=0):
        if count_words(text) <= max_words or level >= len(split_by):
            return [text.strip()]

        separator = split_by[level]
        parts = text.split(separator)
        chunks = []
        current_chunk = ""

        for part in parts:
            if not part.strip():
                continue

            new_chunk = f"{current_chunk}{separator}{part}".strip() if current_chunk else part.strip()

            if count_words(new_chunk) <= max_words:
                current_chunk = new_chunk
            else:
                if current_chunk:
                    chunks.extend(split_text(current_chunk, level + 1))
                current_chunk = part.strip()

        if current_chunk:
            chunks.extend(split_text(current_chunk, level + 1))

        return chunks

    raw_chunks = split_text(text)

    final_chunks = []
    for i in range(len(raw_chunks)):
        chunk = raw_chunks[i]
        final_chunks.append(chunk)

        if i < len(raw_chunks) - 1:
            last_words = " ".join(chunk.split()[-overlap_words:])
            next_words = " ".join(raw_chunks[i + 1].split()[:overlap_words])
            bridge = f"{last_words} {next_words}"
            final_chunks.append(bridge)

    return final_chunks

if __name__ == "__main__":
    with open("./text_document/all_pdf.txt", "r", encoding="utf-8") as f:
        big_text = f.read()

    chunks = recursive_chunk(big_text, max_words=50, overlap_words=100)

    print(f"Total chunks: {len(chunks)}")
    for i, chunk in enumerate(chunks[:6]):
        print(f"\n--- Chunk {i+1} ---\n{chunk[:300]}...")
