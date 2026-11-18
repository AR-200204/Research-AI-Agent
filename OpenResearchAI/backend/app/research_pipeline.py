
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.parse import urlparse

from search_web import search_web
from search_arxiv import search_academic
from extract_webpage import extract_webpage
from extract_pdf import extract_pdf


APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parents[1]
RESEARCH_DIR = PROJECT_DIR / "research_data"
RESEARCH_DIR.mkdir(parents=True, exist_ok=True)


def is_pdf_url(url):
    path = urlparse(url).path.lower()
    return path.endswith(".pdf") or "/pdf/" in path


def is_likely_metadata_page(content, url):
    """
    Detect OpenAlex metadata pages that do not contain
    the actual research paper.
    """
    domain = urlparse(url).netloc.lower()

    if "openalex.org" in domain:
        return True

    return False


def collect_sources(question, web_limit=3, academic_limit=3):
    print("\n" + "=" * 70)
    print("OPENRESEARCH AI - RESEARCH PIPELINE")
    print("=" * 70)
    print(f"\nResearch question: {question}")

    print("\n[STEP 1/4] Searching the web...")
    web_results = search_web(question, max_results=web_limit)

    print("\n[STEP 2/4] Searching academic papers...")
    academic_results = search_academic(
        question,
        max_results=academic_limit
    )

    print("\nWeb results:", len(web_results))
    print("Academic results:", len(academic_results))

    print("\n[STEP 3/4] Preparing research sources...")

    candidates = []

    # Add web search results
    for result in web_results:
        if result.get("url"):
            candidates.append({
                "source_type": "web",
                "title": result.get("title", "Untitled webpage"),
                "url": result["url"],
                "snippet": result.get("snippet", ""),
                "doi": None,
                "year": None,
                "authors": [],
                "is_open_access": None
            })

    # Add academic search results
    for paper in academic_results:
        pdf_url = paper.get("pdf_url")
        paper_url = paper.get("url")

        selected_url = pdf_url or paper_url

        if selected_url:
            candidates.append({
                "source_type": "academic",
                "title": paper.get("title", "Untitled paper"),
                "url": selected_url,
                "paper_url": paper_url,
                "snippet": paper.get("abstract", ""),
                "doi": paper.get("doi"),
                "year": paper.get("year"),
                "authors": paper.get("authors", []),
                "is_open_access": paper.get("is_open_access")
            })

    # Remove duplicate URLs
    unique_candidates = []
    seen_urls = set()

    for candidate in candidates:
        url = candidate["url"]

        if url not in seen_urls:
            seen_urls.add(url)
            unique_candidates.append(candidate)

    print(f"\nUnique sources collected: {len(unique_candidates)}")
    print("\n[STEP 4/4] Extracting source content...")

    collected_sources = []

    for index, candidate in enumerate(unique_candidates, start=1):

        print("\n" + "-" * 70)
        print(f"SOURCE {index}/{len(unique_candidates)}")
        print(f"Title: {candidate['title']}")
        print(f"URL: {candidate['url']}")

        content = ""
        extraction_status = "failed"

        # Extract PDF or webpage
        if is_pdf_url(candidate["url"]):
            print("Source type: PDF")

            result = extract_pdf(
                candidate["url"],
                max_chars=20000
            )

        else:
            print("Source type: HTML")

            result = extract_webpage(
                candidate["url"],
                max_chars=12000
            )

        if result:
            extracted_content = result.get("content", "").strip()

            if extracted_content:

                if is_likely_metadata_page(
                    extracted_content,
                    candidate["url"]
                ):
                    content = extracted_content
                    extraction_status = "metadata_only"

                    print(
                        "Metadata page detected. "
                        "This is not the full research paper."
                    )

                else:
                    content = extracted_content
                    extraction_status = "success"

                    print("Content extraction successful.")

            else:
                print("No readable content extracted.")

        else:
            print("Content extraction failed.")

        # Use search snippet or abstract as fallback
        if not content:
            content = candidate.get("snippet", "").strip()

            if content:
                extraction_status = "snippet_only"
                print("Using search snippet or abstract instead.")

        source = {
            "source_id": f"S{index:03d}",
            "source_type": candidate["source_type"],
            "title": candidate["title"],
            "url": candidate["url"],
            "paper_url": candidate.get("paper_url"),
            "doi": candidate.get("doi"),
            "year": candidate.get("year"),
            "authors": candidate.get("authors", []),
            "is_open_access": candidate.get("is_open_access"),
            "extraction_status": extraction_status,
            "content": content
        }

        collected_sources.append(source)

        print(f"Extraction status: {extraction_status}")
        print(f"Content length: {len(content)} characters")

    return collected_sources


def save_research(question, sources):

    timestamp = datetime.now(timezone.utc)

    research_data = {
        "research_question": question,
        "created_at": timestamp.isoformat(),
        "total_sources": len(sources),

        "successful_extractions": sum(
            1 for source in sources
            if source["extraction_status"] == "success"
        ),

        "metadata_only_sources": sum(
            1 for source in sources
            if source["extraction_status"] == "metadata_only"
        ),

        "snippet_only_sources": sum(
            1 for source in sources
            if source["extraction_status"] == "snippet_only"
        ),

        "failed_extractions": sum(
            1 for source in sources
            if source["extraction_status"] == "failed"
        ),

        "sources": sources
    }

    filename = (
        "research_"
        + timestamp.strftime("%Y%m%d_%H%M%S")
        + ".json"
    )

    output_path = RESEARCH_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            research_data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("RESEARCH COLLECTION COMPLETED")
    print("=" * 70)

    print(f"Research question: {question}")
    print(f"Total sources: {len(sources)}")
    print(
        "Successful full-text extractions:",
        research_data["successful_extractions"]
    )
    print(
        "Metadata-only sources:",
        research_data["metadata_only_sources"]
    )
    print(
        "Snippet-only sources:",
        research_data["snippet_only_sources"]
    )
    print(
        "Failed extractions:",
        research_data["failed_extractions"]
    )

    print(f"\nSaved file: {output_path}")

    print("\nSOURCE SUMMARY")
    print("-" * 70)

    for source in sources:
        print(
            f"{source['source_id']} | "
            f"{source['source_type']} | "
            f"{source['extraction_status']} | "
            f"{source['title']}"
        )

    return output_path


def run_research(question):

    sources = collect_sources(question)

    if not sources:
        print("\nNo research sources were found.")
        return

    save_research(question, sources)


if __name__ == "__main__":

    print("=" * 60)
    print("WELCOME TO OPENRESEARCH AI")
    print("UNIFIED RESEARCH PIPELINE")
    print("=" * 60)

    research_question = input(
        "\nEnter your research topic: "
    ).strip()

    if research_question:
        run_research(research_question)
    else:
        print("Please enter a valid research topic.")
