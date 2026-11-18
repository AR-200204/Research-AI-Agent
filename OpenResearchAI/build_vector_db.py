
import json
import time
from pathlib import Path

import chromadb
import requests


# Project paths
ROOT = Path(__file__).resolve().parent

RESEARCH_FILE = (
    ROOT / "research_data" /
    "research_20261118_062142.json"
)

CHROMA_PATH = ROOT / "chroma_db"

# Models and settings
EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434/api/embed"

CHUNK_SIZE = 1400
CHUNK_OVERLAP = 200
BATCH_SIZE = 16

COLLECTION_NAME = "research_chunks"


def split_text(text):
    """Split text into overlapping chunks."""

    text = " ".join(text.split())

    chunks = []
    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if len(chunk) >= 100:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def generate_embeddings(texts):
    """Generate embeddings using local Ollama."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": EMBEDDING_MODEL,
            "input": texts
        },
        timeout=180
    )

    response.raise_for_status()

    result = response.json()

    return result["embeddings"]


def main():

    print("\nOPENRESEARCH AI - VECTOR DATABASE BUILDER")
    print("=" * 50)

    if not RESEARCH_FILE.exists():
        raise FileNotFoundError(
            f"Research JSON not found: {RESEARCH_FILE}"
        )

    print("\nLoading research data...")

    with open(RESEARCH_FILE, "r", encoding="utf-8") as file:
        research = json.load(file)

    sources = research.get("sources", [])

    documents = []
    metadatas = []
    ids = []

    print(f"Found {len(sources)} sources.")

    # Process each source
    for source in sources:

        source_id = str(source.get("source_id", "UNKNOWN"))
        title = str(source.get("title", "Unknown title"))

        content = source.get("content", "")

        if not isinstance(content, str):
            continue

        content = content.strip()

        status = str(
            source.get("extraction_status", "")
        ).lower()

        # Exclude metadata-only or insufficient content
        if status == "metadata_only" or len(content) < 1000:
            print(f"\nSkipping {source_id}: insufficient content")
            continue

        chunks = split_text(content)

        print(
            f"\n{source_id}: {title}"
            f"\n  Original characters: {len(content)}"
            f"\n  Generated chunks: {len(chunks)}"
        )

        for index, chunk in enumerate(chunks):

            chunk_id = f"{source_id}_chunk_{index + 1:04d}"

            source_type = str(
                source.get("source_type", "unknown")
            )

            # Keep podcast material clearly identifiable
            if "podcast" in (
                source_type + " " + title
            ).lower():
                source_type = "podcast_context"

            metadata = {
                "source_id": source_id,
                "title": title,
                "url": str(
                    source.get("url")
                    or source.get("paper_url")
                    or ""
                ),
                "doi": str(source.get("doi") or ""),
                "year": str(source.get("year") or ""),
                "authors": str(source.get("authors") or ""),
                "source_type": source_type,
                "extraction_status": status,
                "chunk_index": index + 1
            }

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(metadata)

    print("\n" + "=" * 50)
    print(f"Total chunks prepared: {len(documents)}")

    if not documents:
        print("No usable chunks found. Nothing to store.")
        return

    # Connect to persistent ChromaDB
    print("\nConnecting to ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    # Generate embeddings and store in batches
    total = len(documents)

    for start in range(0, total, BATCH_SIZE):

        end = min(start + BATCH_SIZE, total)

        batch_documents = documents[start:end]
        batch_ids = ids[start:end]
        batch_metadata = metadatas[start:end]

        print(
            f"\nEmbedding and storing chunks "
            f"{start + 1}-{end} of {total}..."
        )

        embeddings = generate_embeddings(batch_documents)

        collection.upsert(
            ids=batch_ids,
            documents=batch_documents,
            metadatas=batch_metadata,
            embeddings=embeddings
        )

        print(f"Stored {end}/{total} chunks.")

    print("\n" + "=" * 50)
    print("VECTOR DATABASE BUILD COMPLETED")
    print("=" * 50)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Stored chunks: {collection.count()}")
    print(f"Database location: {CHROMA_PATH}")


if __name__ == "__main__":

    try:
        main()

    except requests.exceptions.ConnectionError:
        print(
            "\nERROR: Cannot connect to Ollama."
            "\nMake sure Ollama is running."
        )

    except Exception as error:
        print(f"\nERROR: {error}")
