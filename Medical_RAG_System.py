#!/usr/bin/env python3
"""
"""

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
from typing import List
from urllib.parse import urljoin, urlparse
from tqdm import tqdm 
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings

os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36")

def get_links_from_hub(hub_url: str, filter_keyword: str) -> List[str]:
    """Scans a hub page for all sub-links matching the filter."""
    print(f"--- Scanning hub page for URLs: {hub_url} ---")
    try:
        r = requests.get(hub_url, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        
        links = []
        for a in soup.find_all('a', href=True):
            full_url = urljoin(hub_url, a['href']).split('#')[0]
            
            # Stay on domain and check for specific WHO fact sheet path
            if urlparse(full_url).netloc == urlparse(hub_url).netloc:
                if filter_keyword in full_url:
                    links.append(full_url)
        
        unique_links = sorted(list(set(links)))
        print(f"Found {len(unique_links)} unique URLs to process.")
        return unique_links
    except Exception as e:
        print(f"Error scanning hub: {e}")
        return []

def load_web_html(url: str) -> List[Document]:
    """Downloads and extracts full webpage text."""
    try:
        r = requests.get(url, timeout=15)
        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.extract()

        # WHO fact sheets usually use an 'article' or 'div.sf-content-block' tag
        content = soup.find("article") or soup.find("main") or soup.find("div", class_="sf-content-block")
        if content:
            soup = content

        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned_text = "\n".join(lines)

        return [Document(page_content=cleaned_text, metadata={"source": url})]
    except Exception as e:
        return []


HUB_URL = "https://www.who.int/news-room/fact-sheets"
FILTER = "/news-room/fact-sheets/detail/" 

target_urls = get_links_from_hub(HUB_URL, FILTER)

if not target_urls:
    print("\n[!] CRITICAL ERROR: No URLs found. Check your FILTER or internet connection.")
    exit()

target_urls = target_urls[:15]

raw_docs = []
print(f"\n--- Starting to scrape {len(target_urls)} pages ---")
for u in tqdm(target_urls, desc="Reading WHO Pages"):
    raw_docs.extend(load_web_html(u))

if not raw_docs:
    print("[!] ERROR: No content could be scraped from the found URLs.")
    exit()

print(f"\nTotal documents loaded into system: {len(raw_docs)}")

# Chunking 

def paragraph_chunker(docs):
    return RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150, separators=[". ", "? ", "! ", "\n"]).split_documents(docs)

# --- LLM Initialization ---
print("\nInitializing Ollama...")
emb = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="mistral")

def call_llm(prompt: str) -> str:
    out = llm.invoke(prompt)
    return out.content if hasattr(out, 'content') else str(out)

PROMPT_TEMPLATE = """
You are a medical health expert. Use ONLY the provided context to answer. 
If the answer is not found, reply strictly with: "I don't know (not found in context)."

Context:
{context}

Question:
{question}

Answer:
"""

def answer_with_rag(db, question: str, k: int = 4):
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in docs])
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)
    return call_llm(prompt), docs

QUESTIONS = [
    "How many unintended pregnancies end in induced abortion?",
    "Who is Anemia a major public health concern for?",
    "What is Autism?",
    "How many people are bitten by snakes worldwide every year?",
    "How many deaths were caused by alcohol consumption in 2019?"
]

# --- Processing ---
results = []
chunkers = {
    "paragraph": paragraph_chunker(raw_docs)
}

for name, chunks in chunkers.items():
    print(f"\n--- Building FAISS for {name} chunker ({len(chunks)} chunks) ---")
    db = FAISS.from_documents(chunks, emb)

    for q in QUESTIONS:
        ans, docs = answer_with_rag(db, q)
        print(f"[{name}] Q: {q}\nA: {ans[:200]}...\n")
        results.append({
            "chunker": name,
            "question": q,
            "answer": ans,
            "num_chunks": len(docs)
        })

# Save results
pd.DataFrame(results).to_csv("RAG_results.csv", index=False)
print("\nSuccess! Results saved.")