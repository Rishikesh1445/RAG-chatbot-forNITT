import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse

BASE_URL = "https://www.nitt.edu/"
OUTPUT_DIR = "pdfs"
visited = set()
pdf_links = set()

os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def crawl(url):
    if url in visited:
        return
    visited.add(url)

    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if "text/html" not in r.headers.get("Content-Type", ""):
            return
    except:
        return

    if "<html" not in r.text.lower():
        return

    soup = BeautifulSoup(r.text, "html.parser")

    for a in soup.find_all("a", href=True):
        link = urljoin(url, a["href"])
        link_lower = link.lower()

        if ".pdf" in link_lower:
            pdf_links.add(link)

        elif link.startswith(BASE_URL):
            crawl(link)

crawl(BASE_URL)

print("PDFs found:", len(pdf_links))

for pdf_url in pdf_links:
    filename = os.path.basename(urlparse(pdf_url).path)
    if not filename:
        continue

    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        continue

    try:
        pdf = requests.get(pdf_url, headers=HEADERS, timeout=20)
        with open(path, "wb") as f:
            f.write(pdf.content)
        print("Downloaded:", filename)
    except:
        pass
