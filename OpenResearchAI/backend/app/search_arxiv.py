
import os
import requests

from pathlib import Path
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BACKEND_DIR = Path(__file__).resolve().parents[1]

load_dotenv(BACKEND_DIR / ".env")

API_URL = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index):

    if not inverted_index:
        return "Abstract unavailable"

    words = {}

    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word

    return " ".join(
        words[position]
        for position in sorted(words)
    )


def create_session():

    session = requests.Session()

    retries = Retry(
        total=2,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        respect_retry_after_header=True
    )

    adapter = HTTPAdapter(max_retries=retries)

    session.mount("https://", adapter)

    return session


def search_academic(query, max_results=5):

    api_key = os.getenv("OPENALEX_API_KEY")

    if not api_key:
        print("ERROR: OPENALEX_API_KEY not found in backend/.env")
        return []

    print(f"\nSearching OpenAlex for: {query}\n")

    params = {
        "search": query,
        "per_page": max_results,
        "api_key": api_key,
        "select": (
            "id,doi,title,publication_year,"
            "authorships,abstract_inverted_index,"
            "primary_location,best_oa_location,"
            "open_access"
        )
    }

    try:

        session = create_session()

        response = session.get(
            API_URL,
            params=params,
            timeout=(15, 60)
        )

        response.raise_for_status()

        data = response.json()
        works = data.get("results", [])

        if not works:
            print("No academic papers found.")
            return []

        papers = []

        for index, work in enumerate(works, start=1):

            authors = []

            for authorship in work.get("authorships", []):
                author = authorship.get("author", {})
                name = author.get("display_name")

                if name:
                    authors.append(name)

            abstract = reconstruct_abstract(
                work.get("abstract_inverted_index")
            )

            open_access = work.get("open_access") or {}
            best_location = work.get("best_oa_location") or {}

            paper = {
                "number": index,
                "title": work.get("title") or "No title",
                "authors": authors,
                "year": work.get("publication_year"),
                "abstract": abstract,
                "url": work.get("id", ""),
                "doi": work.get("doi"),
                "pdf_url": best_location.get("pdf_url"),
                "is_open_access": open_access.get("is_oa", False)
            }

            papers.append(paper)

            print("=" * 70)
            print(f"[{index}] {paper['title']}")
            print(f"Authors: {', '.join(authors[:5])}")
            print(f"Year: {paper['year']}")
            print(f"Open access: {paper['is_open_access']}")
            print(f"OpenAlex: {paper['url']}")
            print(f"DOI: {paper['doi']}")
            print(f"PDF: {paper['pdf_url']}")
            print(f"Abstract: {abstract[:600]}...")

        print("=" * 70)
        print(f"\nRetrieved {len(papers)} academic papers.")

        return papers

    except requests.HTTPError as error:

        print(f"\nOpenAlex HTTP error: {error}")

        if response.status_code == 429:
            print("API rate limit reached. Wait before trying again.")

        elif response.status_code in (401, 403):
            print("Check whether your API key is valid and activated.")

        return []

    except requests.RequestException as error:
        print(f"\nOpenAlex connection error: {error}")
        return []

    except ValueError as error:
        print(f"\nCould not read API response: {error}")
        return []


if __name__ == "__main__":

    question = input(
        "Enter an academic research topic: "
    ).strip()

    if question:
        search_academic(question)
    else:
        print("Please enter a research topic.")
