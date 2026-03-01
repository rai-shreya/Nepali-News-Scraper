import asyncio
import aiohttp
from bs4 import BeautifulSoup
import json
from datetime import datetime
from urllib.parse import urljoin
import hashlib
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings

DATA_FILE = "nepali_news_rag.json"
PERSIST_DIR = "chroma_db"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}
MAX_WORDS_PER_CHUNK = 500

SITES = [
    ("Kathmandu Post", "https://kathmandupost.com/politics", "https://kathmandupost.com"),
    ("Online Khabar", "https://english.onlinekhabar.com/category/political", ""),
    ("Setopati", "https://en.setopati.com/political", ""),
    ("Nepali Press", "https://english.nepalpress.com/category/political/", ""),
    ("eKantipur", "https://ekantipur.com/politics", "https://ekantipur.com"),
]


def clean_text(text):
    import re
    text = text.replace("\n", " ").strip()
    return re.sub(r"\s+", " ", text)

def chunk_text(text, max_len=MAX_WORDS_PER_CHUNK):
    words = text.split()
    return [" ".join(words[i:i+max_len]) for i in range(0, len(words), max_len)]

def content_hash(text):
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def load_existing():
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


async def fetch(session, url):
    try:
        async with session.get(url, headers=HEADERS, timeout=15) as r:
            if r.status != 200:
                print(f" Failed: {url} -> {r.status}")
                return None
            return await r.text()
    except Exception as e:
        print(f" Exception fetching {url}: {e}")
        return None

async def scrape_article(session, url, source):
    html = await fetch(session, url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    paragraphs = soup.find_all("p")
    content = clean_text(" ".join(p.get_text() for p in paragraphs))

    if len(content) < 300:
        return []

    chunks = chunk_text(content)
    data = []
    for chunk in chunks:
        data.append({
            "source": source,
            "url": url,
            "content": chunk,
            "scraped_at": datetime.utcnow().isoformat(),
            "hash": content_hash(chunk)
        })
    return data

async def scrape_site(session, source, list_url, base_url=""):
    html = await fetch(session, list_url)
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    anchors = soup.find_all("a")
    urls = set()

    for a in anchors:
        href = a.get("href")
        if not href:
            continue
        url = urljoin(base_url, href)
        if any(keyword in url for keyword in ["202", "politic", "news"]):
            urls.add(url)

    print(f" {source}: {len(urls)} article links found")

    tasks = [scrape_article(session, url, source) for url in urls]
    results = await asyncio.gather(*tasks)
    all_chunks = [chunk for sublist in results for chunk in sublist]
    print(f" {len(all_chunks)} chunks scraped from {source}")
    return all_chunks


async def main():
    existing = load_existing()
    existing_hashes = set(a["hash"] for a in existing)

    async with aiohttp.ClientSession() as session:
        tasks = [scrape_site(session, *site) for site in SITES]
        results = await asyncio.gather(*tasks)

    all_chunks = [item for sublist in results for item in sublist]
    new_chunks = [c for c in all_chunks if c["hash"] not in existing_hashes]

    if new_chunks:
        combined = existing + new_chunks
        save_data(combined)
        print(f"\n Saved {len(new_chunks)} new chunks. Total: {len(combined)}")
    else:
        print("\n⚠ No new chunks found.")
        combined = existing

    print(" Generating embeddings and storing in Chroma...")
    embedding = OllamaEmbeddings(model="nomic-embed-text")
    texts = [c["content"] for c in combined]
    metadatas = [{"source": c["source"], "url": c["url"], "scraped_at": c["scraped_at"]} for c in combined]

    vectordb = Chroma.from_texts(texts=texts, embedding=embedding, metadatas=metadatas, persist_directory=PERSIST_DIR)
    vectordb.persist()
    print(" Chroma DB updated. Pipeline complete!")

if __name__ == "__main__":
    asyncio.run(main())