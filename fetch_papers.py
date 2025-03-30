import arxiv
import requests
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor

def fetch_arxiv_papers(query, max_results=5):
    """Fetches research papers from ArXiv based on a query."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    papers = []
    for result in client.results(search):
        print(result.title)  # Print the title of each paper
        papers.append({
            "title": result.title,
            "abstract": result.summary,
            "url": result.entry_id,
            "pdf_url": result.pdf_url,
            "source": "arxiv"
        })

    return papers

def extract_text_from_pdf(pdf_url):
    """Downloads a PDF and extracts text in-memory without saving to disk."""
    response = requests.get(pdf_url, stream=True, timeout=10)  # Stream for faster response
    if response.status_code == 200:
        doc = fitz.open("pdf", response.content)  # Open PDF from bytes
        text = "\n".join(page.get_text() for page in doc)
        return text
    return None  # Return None if extraction fails

def process_paper(paper):
    """Processes a single research paper (downloads and extracts text)."""
    pdf_text = extract_text_from_pdf(paper["pdf_url"])  # Extract text

    abstract_text = paper["abstract"]

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=200)
    text_chunks = text_splitter.split_text(abstract_text)
    
    return {
        "title": paper["title"],
        "abstract" : paper["abstract"],
        "chunks": text_chunks,
        "pdf_url": paper["pdf_url"],
        "source": paper.get("source","unknown"),
        "full_text": pdf_text or "PDF could not be processed."
    }

def get_research_papers(query, max_results=5):
    """Fetches and processes research papers in parallel for efficiency."""
    papers = fetch_arxiv_papers(query, max_results)
    
    # Parallelize PDF extraction
    with ThreadPoolExecutor(max_workers=5) as executor:
        processed_papers = list(executor.map(process_paper, papers))

    return processed_papers