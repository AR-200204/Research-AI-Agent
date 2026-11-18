
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def extract_webpage(url, max_chars=12000):
    """
    Download a webpage and extract its readable text.

    Args:
        url: Webpage URL.
        max_chars: Maximum number of characters to extract.

    Returns:
        A dictionary containing the page title, URL,
        description and extracted text.
    """

    print(f"\nExtracting webpage: {url}\n")

    # Validate URL
    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        print("Invalid URL. Please provide an HTTP or HTTPS URL.")
        return None

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(10, 30)
        )

        response.raise_for_status()

        # Check whether the URL returns HTML
        content_type = response.headers.get("Content-Type", "")

        if "text/html" not in content_type.lower():
            print("This URL does not return an HTML webpage.")
            print(f"Content type: {content_type}")
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        # Extract page title
        title = (
            soup.title.get_text(strip=True)
            if soup.title
            else "No title found"
        )

        # Extract meta description
        description = ""

        meta_description = soup.find(
            "meta",
            attrs={"name": "description"}
        )

        if meta_description:
            description = meta_description.get("content", "")

        # Remove unnecessary HTML elements
        unwanted_elements = [
            "script",
            "style",
            "nav",
            "footer",
            "header",
            "aside",
            "form",
            "noscript",
            "iframe"
        ]

        for element in soup(unwanted_elements):
            element.decompose()

        # Prefer the main article content when available
        main_content = soup.find("article") or soup.find("main")

        if main_content:
            text = main_content.get_text(
                separator=" ",
                strip=True
            )
        else:
            text = soup.get_text(
                separator=" ",
                strip=True
            )

        # Clean unnecessary whitespace
        text = " ".join(text.split())

        if not text:
            print("No readable text was found.")
            return None

        # Limit extracted content
        original_length = len(text)
        text = text[:max_chars]

        result = {
            "title": title,
            "url": response.url,
            "description": description,
            "content": text,
            "original_length": original_length,
            "extracted_length": len(text)
        }

        print("=" * 70)
        print("WEBPAGE EXTRACTION SUCCESSFUL")
        print("=" * 70)

        print(f"Title: {title}")
        print(f"URL: {response.url}")
        print(f"Original text length: {original_length}")
        print(f"Extracted text length: {len(text)}")

        if original_length > max_chars:
            print("Note: Content was shortened to the character limit.")

        print("\nEXTRACTED CONTENT")
        print("-" * 70)
        print(text[:2000])

        print("\n" + "=" * 70)

        return result

    except requests.exceptions.Timeout:
        print("Error: The webpage took too long to respond.")

    except requests.exceptions.HTTPError as error:
        print(f"HTTP error: {error}")

    except requests.exceptions.RequestException as error:
        print(f"Connection error: {error}")

    except Exception as error:
        print(f"Unexpected error: {error}")

    return None


if __name__ == "__main__":

    print("=" * 60)
    print("OPENRESEARCH AI - WEBPAGE EXTRACTOR")
    print("=" * 60)

    webpage_url = input("\nEnter a webpage URL: ").strip()

    if webpage_url:
        extract_webpage(webpage_url)
    else:
        print("Please enter a valid URL.")
