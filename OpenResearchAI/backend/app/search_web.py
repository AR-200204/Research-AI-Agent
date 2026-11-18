
import os
from pathlib import Path

from dotenv import load_dotenv
from tavily import TavilyClient


# Load the .env file from the backend directory
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def search_web(query, max_results=5):
    """
    Search the internet using Tavily.
    Returns structured search results.
    """

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        print("ERROR: TAVILY_API_KEY was not found.")
        return []

    print(f"\nSearching the web for: {query}\n")

    try:
        client = TavilyClient(api_key=api_key)

        response = client.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )

        results = response.get("results", [])
        search_results = []

        if not results:
            print("No search results found.")
            return []

        for index, result in enumerate(results, start=1):

            item = {
                "number": index,
                "title": result.get("title", "No title"),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
                "score": result.get("score", 0)
            }

            search_results.append(item)

            print("=" * 70)
            print(f"[{index}] {item['title']}")
            print(f"URL: {item['url']}")
            print(f"Relevance: {item['score']:.3f}")
            print(f"Content: {item['snippet']}")

        print("=" * 70)
        print(f"\nSuccessfully retrieved {len(search_results)} results.")

        return search_results

    except Exception as error:
        print("\nTavily search failed:")
        print(f"{type(error).__name__}: {error}")
        return []


if __name__ == "__main__":

    question = input("Enter a topic to research: ").strip()

    if question:
        search_web(question)
    else:
        print("Please enter a search topic.")
