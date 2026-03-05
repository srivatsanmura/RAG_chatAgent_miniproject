import trafilatura
from rag.logger import logger
import os
import hashlib
import re
import time

MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles on each retry)

def extract_url(url: str):
    """
    Extracts the text from a given URL using trafilatura.
    Retries the download up to MAX_RETRIES times with exponential backoff.

    Args:
        url (str): The URL to extract text from.

    Returns:
        dict: A dictionary containing the extracted text and the source URL.
    """
    logger.info(f"Extracting: {url}")

    # --- Download with retry ---
    downloaded = None
    for attempt in range(1, MAX_RETRIES + 1):
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            logger.info(f"Downloaded: {url} (attempt {attempt})")
            break
        delay = RETRY_DELAY * (2 ** (attempt - 1))  # 2s, 4s, 8s
        logger.warning(f"Download attempt {attempt} failed for: {url}. Retrying in {delay}s...")
        time.sleep(delay)

    if not downloaded:
        logger.error(f"Failed to download after {MAX_RETRIES} attempts: {url}")
        return None

    # --- Extract text ---
    text = trafilatura.extract(downloaded)

    if not text:
        logger.error(f"Failed to extract text from: {url}")
        return None

    logger.info(f"Extracted: {url}")

    # --- Save to disk ---
    title = None
    try:
        os.makedirs("data/raw", exist_ok=True)

        m = trafilatura.extract_metadata(downloaded)
        if m and getattr(m, 'title', None):
            title = m.title
            filename = re.sub(r'[\s\-]+', '_', title)
            # Remove invalid characters for Windows filenames
            filename = re.sub(r'[\\/*?:"<>|]', '', filename)
            filename = f"{filename}.txt"
        else:
            filename = hashlib.md5(url.encode('utf-8')).hexdigest() + ".txt"

        file_path = os.path.join("data/raw", filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"Saved extracted content to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save content to disk: {e}")

    return {
        "content": text,
        "title": title,
        "source": url
    }


def extract_urls(urls):
    """
    Ingests multiple URLs and extracts text from each.

    Args:
        urls (list): A list of URLs to ingest.

    Returns:
        list: A list of dictionaries containing the extracted text and source URLs.
    """

    documents = []

    for url in urls:    

        doc = extract_url(url)

        if doc:
            documents.append(doc)

    logger.info(f"Ingested {len(documents)} docs")

    return documents