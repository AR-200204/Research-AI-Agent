
import chromadb
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent

client = chromadb.PersistentClient(
    path=str(ROOT / "chroma_db")
)

collection = client.get_collection(
    name="research_chunks"
)

def search_research(question, top_k=5):

    response = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": question
        },
        timeout=120
    )

    response.raise_for_status()

    query_embedding = response.json()["embeddings"][0]

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    print("\nQUESTION:", question)
    print("=" * 70)

    for i, (document, metadata, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        print(f"\nRESULT {i}")
        print("-" * 70)
        print("Source:", metadata["source_id"])
        print("Title:", metadata["title"])
        print("Chunk:", metadata["chunk_index"])
        print("Distance:", round(distance, 4))
        print("\nRetrieved passage:")
        print(document[:700])
        print("\n")

    print("=" * 70)


if __name__ == "__main__":

    print("OPENRESEARCH AI - SEMANTIC SEARCH")
    print("Available chunks:", collection.count())

    question = input("\nEnter your research question: ")

    if question.strip():
        search_research(question)
