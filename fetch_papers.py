import arxiv
import requests
import fitz  # PyMuPDF
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from concurrent.futures import ThreadPoolExecutor
from setup_mongo import db,collection  # Import the MongoDB collection
import re

def fetch_arxiv_papers(query, max_results=5):
    """Fetches research papers from ArXiv based on a query."""
    client = arxiv.Client()
    search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance)

    papers = []
    for result in client.results(search):
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
    try:
        response = requests.get(pdf_url, stream=True, timeout=10)  # Stream for faster response
        if response.status_code == 200:
            doc = fitz.open("pdf", response.content)  # Open PDF from bytes
            text = "\n".join(page.get_text() for page in doc)
            return text
    except Exception as e:
        print(f"⚠️ Error processing {pdf_url}: {str(e)}")
        return None  # Return None if extraction fails


def extract_citations(full_text):
    """Extract arXiv DOIs from full text."""
    arxiv_pattern = r'arXiv:\d{4}\.\d{4,5}'  
    citations = re.findall(arxiv_pattern, full_text)
    return citations

def process_paper(paper):
    """Processes a single research paper (downloads and extracts text)."""
    pdf_text = extract_text_from_pdf(paper["pdf_url"])  # Extract text

    # remove citations from the full_text and white spaces using regex    
    # improvised_full_text_without_citations = re.sub(r'\([A-Za-z]+\s?\d{4}\)', ' ', pdf_text)
    # improvised_full_text_without_citations= re.sub(r'\b\d{4}[a-z]?\b', '', improvised_full_text_without_citations)
    
    # Combine relevant content for chunking
    content = f"Title: {paper['title']}\nAbstract: {paper['abstract']}\n"

    # extract citations
    # citations = extract_citations(pdf_text)

    

    if pdf_text:
        content += f"Full Text: {pdf_text}"

    # abstract_text = paper["abstract"]

    # Split into chunks
    text_chunks=[]
    if pdf_text:
        text_splitter = RecursiveCharacterTextSplitter( chunk_size=1200, 
                                                        chunk_overlap=240,
                                                        separators= ["\n\n", "\n", " ", ".",""],
                                                        length_function=len,
                                                        is_separator_regex=False)
        text_chunks = [chunk for chunk in text_splitter.split_text(content) if len(chunk.strip()) > 100]

    return {
        "title": paper["title"],
        "abstract" : paper["abstract"],
        # "chunks": text_chunks,
        "pdf_url": paper["pdf_url"],
        "source": paper.get("source","unknown"),
        "text_chunks": text_chunks,
        # "citations" : citations or [],  # List of citations extracted from the full text
    }

def get_research_papers(query, max_results=5):
    """Fetches and processes research papers in parallel for efficiency."""
    papers = fetch_arxiv_papers(query, max_results)
    
    existing_pdfs = set(doc["pdf_url"] for doc in collection.find({}, {"pdf_url": 1}))
    new_papers = [paper for paper in papers if paper["pdf_url"] not in existing_pdfs]
    if new_papers:
        # Parallelize PDF extraction
        with ThreadPoolExecutor(max_workers=5) as executor:
            processed_papers = list(executor.map(process_paper, new_papers))

        if processed_papers:
            # Insert new papers into MongoDB
            collection.insert_many(processed_papers)  # Only insert new ones
            print(f"✅ Inserted/Updated {len(processed_papers)} papers into MongoDB.")

        # Step 4: Return all papers but exclude text_chunks
        # Get all papers from MongoDB that match the fetched ones
    matched_papers = list(collection.find(
        {"pdf_url": {"$in": [paper["pdf_url"] for paper in papers]}},  # Only return matching papers
        {"_id":0 , "text_chunks": 0}  # Exclude text_chunks
    )) 

    return matched_papers


