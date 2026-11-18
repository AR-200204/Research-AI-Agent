
import re
import chromadb
import requests
from pathlib import Path


# ============================================================
# OPENRESEARCH AI - RAG RESEARCH ASSISTANT
# ============================================================

ROOT = Path(__file__).resolve().parent

CHROMA_PATH = ROOT / "chroma_db"

EMBEDDING_MODEL = "nomic-embed-text"
CHAT_MODEL = "qwen3:4b-instruct"

OLLAMA_URL = "http://localhost:11434"

TOP_K = 3


# ============================================================
# GENERATE QUESTION EMBEDDING
# ============================================================

def generate_embedding(question):

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": question
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["embeddings"][0]


# ============================================================
# RETRIEVE RELEVANT RESEARCH
# ============================================================

def retrieve_research(question, collection):

    embedding = generate_embedding(question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(TOP_K, collection.count()),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):

        retrieved.append({
            "text": document,
            "source_id": metadata["source_id"],
            "title": metadata["title"],
            "chunk": metadata["chunk_index"],
            "distance": distance
        })

    return retrieved


# ============================================================
# VALIDATE AND BLOCK UNSUPPORTED CITATIONS
# ============================================================

def validate_citations(answer, retrieved):

    valid_citations = {
        (item["source_id"], int(item["chunk"]))
        for item in retrieved
    }

    invalid_citations = []

    citation_pattern = (
        r"\[([A-Za-z0-9_-]+),\s*chunk\s*(\d+)\]"
    )

    def check_citation(match):

        source_id = match.group(1)
        chunk_number = int(match.group(2))

        citation = (source_id, chunk_number)

        if citation not in valid_citations:

            invalid_citations.append(citation)

            return "[UNVERIFIED CITATION]"

        return match.group(0)

    # Replace unsupported citations automatically
    answer = re.sub(
        citation_pattern,
        check_citation,
        answer,
        flags=re.IGNORECASE
    )

    if invalid_citations:

        print("\nWARNING: Unsupported citations detected!")

        for source_id, chunk in sorted(set(invalid_citations)):
            print(f"Blocked: [{source_id}, chunk {chunk}]")

    else:

        print("\nCitation validation: PASSED")

    return answer, invalid_citations


# ============================================================
# GENERATE ANSWER USING QWEN
# ============================================================

def generate_answer(question, retrieved):

    context_parts = []

    for i, item in enumerate(retrieved, start=1):

        context_parts.append(
            f"""
EVIDENCE {i}

Source ID: {item['source_id']}
Title: {item['title']}
Chunk: {item['chunk']}

Content:
{item['text']}
"""
        )

    context = "\n".join(context_parts)

    prompt = f"""
You are OpenResearch AI, an academic research assistant.

Answer the research question using ONLY the supplied
research evidence.

RESEARCH QUESTION:
{question}

RETRIEVED RESEARCH EVIDENCE:
{context}

IMPORTANT INSTRUCTIONS:

1. Give a clear, well-organized academic answer.

2. Explain important concepts in understandable language.

3. Use citations in this exact format:
   [S001, chunk 1]

4. ONLY cite source IDs and chunk numbers explicitly
   provided in the retrieved evidence.

5. NEVER invent a citation or refer to a chunk
   that was not retrieved.

6. If the evidence is insufficient, explicitly say so.

7. Distinguish evidence from different sources.

8. Do not claim that all sources are peer-reviewed.

9. Do not use outside knowledge as if it came
   from the supplied sources.

10. Do not generate a separate References section.
    Inline citations are sufficient.

11. Keep the answer concise, approximately
    300-400 words.

12. Write the answer with appropriate headings
    and a complete conclusion.

13. Finish the answer completely.
    Do not leave unfinished sentences.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": CHAT_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_ctx": 2048,
                "num_predict": 650
            }
        },
        timeout=600
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# ============================================================
# DISPLAY RETRIEVED SOURCES
# ============================================================

def display_sources(retrieved):

    print("\n" + "=" * 60)
    print("RETRIEVED SOURCES")
    print("=" * 60)

    for item in retrieved:

        print(
            f"\n[{item['source_id']}, "
            f"chunk {item['chunk']}]"
        )

        print("Title:", item["title"])

        print(
            "Distance:",
            round(item["distance"], 4)
        )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():

    print("\nOPENRESEARCH AI - RAG RESEARCH ASSISTANT")
    print("=" * 60)

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name="research_chunks"
    )

    print("Vector database loaded.")

    print(
        "Available research chunks:",
        collection.count()
    )

    while True:

        question = input(
            "\nEnter your research question "
            "(or type exit): "
        ).strip()

        if question.lower() in ["exit", "quit"]:

            print("\nClosing OpenResearch AI.")
            break

        if not question:
            continue

        try:

            # STEP 1: RETRIEVE RESEARCH
            print("\nSearching research database...")

            retrieved = retrieve_research(
                question,
                collection
            )

            print(
                f"Retrieved {len(retrieved)} research chunks."
            )

            if not retrieved:

                print("No relevant research evidence found.")
                continue

            # STEP 2: GENERATE ANSWER
            print("\nGenerating evidence-based answer...")

            answer = generate_answer(
                question,
                retrieved
            )

            # STEP 3: VALIDATE AND BLOCK CITATIONS
            answer, invalid_citations = validate_citations(
                answer,
                retrieved
            )

            validation = {
                "valid": len(invalid_citations) == 0,
                "invalid_citations": invalid_citations
            }

            # STEP 4: DISPLAY ANSWER
            print("\n" + "=" * 60)
            print("OPENRESEARCH AI - ANSWER")
            print("=" * 60)

            print(answer)

            # STEP 5: DISPLAY VALIDATION RESULT
            print("\n" + "=" * 60)
            print("CITATION VALIDATION")
            print("=" * 60)

            if validation["valid"]:

                print(
                    "All citations reference retrieved chunks."
                )

            else:

                print(
                    "Unsupported citations were blocked."
                )

            # STEP 6: DISPLAY SOURCES
            display_sources(retrieved)

        except requests.exceptions.ConnectionError:

            print(
                "\nERROR: Cannot connect to Ollama."
                "\nMake sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            print(
                "\nERROR: Ollama request timed out."
                "\nTry again with a shorter question."
            )

        except Exception as error:

            print("\nERROR:", error)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
