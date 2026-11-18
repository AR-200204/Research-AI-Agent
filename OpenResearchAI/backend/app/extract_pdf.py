
import requests
import pymupdf as fitz
from urllib.parse import urlparse


def extract_pdf(url, max_chars=20000):
    """
    Download a PDF research paper and extract readable text.

    Args:
        url: PDF URL.
        max_chars: Maximum number of characters to extract.

    Returns:
        Dictionary containing PDF metadata and extracted text.
    """

    print(f"\nDownloading PDF: {url}\n")

    parsed_url = urlparse(url)

    if parsed_url.scheme not in ("http", "https") or not parsed_url.netloc:
        print("Invalid PDF URL.")
        return None

    headers = {
        "User-Agent": "Mozilla/5.0 OpenResearchAI/1.0"
    }

    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=(15, 60)
        )

        response.raise_for_status()

        # Check PDF file size (maximum 30 MB)
        max_file_size = 30 * 1024 * 1024

        if len(response.content) > max_file_size:
            print("PDF is too large. Maximum allowed size is 30 MB.")
            return None

        # Open PDF directly from downloaded memory
        pdf_document = fitz.open(
            stream=response.content,
            filetype="pdf"
        )

        if pdf_document.page_count == 0:
            print("The PDF contains no pages.")
            return None

        title = pdf_document.metadata.get("title") or "Unknown title"
        author = pdf_document.metadata.get("author") or "Unknown author"

        print("=" * 70)
        print("PDF EXTRACTION SUCCESSFUL")
        print("=" * 70)

        print(f"Title: {title}")
        print(f"Author: {author}")
        print(f"Total pages: {pdf_document.page_count}")

        extracted_pages = []
        total_length = 0

        for page_number, page in enumerate(pdf_document, start=1):

            page_text = page.get_text("text")
            page_text = " ".join(page_text.split())

            if page_text:
                extracted_pages.append(
                    f"\n[Page {page_number}]\n{page_text}"
                )

                total_length += len(page_text)

        full_text = "\n".join(extracted_pages)

        if not full_text.strip():
            print("No readable text found. This may be a scanned PDF.")
            pdf_document.close()
            return None

        extracted_text = full_text[:max_chars]

        result = {
            "title": title,
            "author": author,
            "url": response.url,
            "total_pages": pdf_document.page_count,
            "original_length": len(full_text),
            "extracted_length": len(extracted_text),
            "content": extracted_text
        }

        pdf_document.close()

        print(f"Original text length: {len(full_text)}")
        print(f"Extracted text length: {len(extracted_text)}")

        if len(full_text) > max_chars:
            print("Note: Content was shortened to the character limit.")

        print("\nEXTRACTED CONTENT")
        print("-" * 70)
        print(extracted_text[:3000])

        print("\n" + "=" * 70)

        return result

    except requests.exceptions.Timeout:
        print("Error: PDF download timed out.")

    except requests.exceptions.HTTPError as error:
        print(f"HTTP error: {error}")

    except requests.exceptions.RequestException as error:
        print(f"Download error: {error}")

    except Exception as error:
        print(f"PDF extraction error: {error}")

    return None


if __name__ == "__main__":

    print("=" * 60)
    print("OPENRESEARCH AI - PDF EXTRACTOR")
    print("=" * 60)

    pdf_url = input("\nEnter a PDF URL: ").strip()

    if pdf_url:
        extract_pdf(pdf_url)
    else:
        print("Please enter a valid PDF URL.")
