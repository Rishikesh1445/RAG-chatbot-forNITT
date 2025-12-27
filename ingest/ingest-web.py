import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re

BASE_URL = "https://www.nitt.edu/"
OUTPUT_DIR = "../data/web"
visited = set()

os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def is_valid_url(url):
    parsed = urlparse(url)
    return parsed.netloc == "www.nitt.edu"

def crawl(url):
    if url in visited:
        return
    visited.add(url)

    # filename + filepath FIRST (needed to skip existing files)
    filename = url.replace("https://www.nitt.edu/", "").replace("/", "_")
    if not filename:
        filename = "home"
    filepath = os.path.join(OUTPUT_DIR, f"{filename}.txt")

    # skip already downloaded pages
    if os.path.exists(filepath):
        return

    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            return

    except:
        return

    if "<html" not in response.text.lower():
        return

    try:
        response.encoding = response.apparent_encoding
        html = response.text
        soup = BeautifulSoup(html, "html.parser")
    except Exception:
        return

    # Remove junk
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    text = clean_text(soup.get_text(separator=" "))

    if len(text) > 500:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

    # Find all internal links
    for link in soup.find_all("a", href=True):
        next_url = urljoin(url, link["href"])
        if is_valid_url(next_url):
            crawl(next_url)

if __name__ == "__main__":
    crawl(BASE_URL)
