from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from fetch_papers import get_research_papers
from langchain.docstore.document import Document  # Import the Document class
import shutil

# Load a Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS database directory
DB_NAME = "research_papers_faiss"


def delete_faiss_db():
    """Deletes the FAISS database if it exists."""
    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)
        print("🗑️ FAISS database deleted.")
    else:
        print("ℹ️ No FAISS database found.")

# Function to create FAISS database from processed papers
def upload_papers_to_faiss(user_query : str ,max_papers : int=5,generate_new :bool =True):
    if not generate_new and os.path.exists(DB_NAME):
        vector_db = FAISS.load_local(DB_NAME, 
                                     embeddings=embedding_model,
                                     allow_dangerous_deserialization=True  # Explicitly enable for trusted sources
                                     )   
        print(f"✅ FAISS database default loaded.")
        return vector_db
    if generate_new:
        delete_faiss_db()  # Delete existing database if generate_new is True

    papers = get_research_papers(user_query,max_papers)  # Get processed papers from fetch_papers.py
    docs = []
    
    for paper in papers:
        print(paper["pdf_url"])
        chunk=paper["title"]+paper["abstract"]
        # for chunk in paper["chunks"]:
            # Create a Document object for each chunk
        docs.append(Document(
            page_content=chunk,  # Chunk text
            metadata={           # Metadata associated with the chunk
                "title": paper["title"],  # Paper's title
                "abstract": paper["abstract"],  # Paper's abstract
                "pdf_url": paper["pdf_url"],  # URL to the PDF
                "source": paper["source"],  # Source of the paper (e.g., arXiv)
                "full_text": paper["full_text"]  # Full text of the paper
            }   
    ))

    if not docs:
        print("❌ No valid text extracted from papers.")
        return None

    # Check if the FAISS database already exists
    if os.path.exists(DB_NAME):
        vector_db = FAISS.load_local(DB_NAME, 
                                        embeddings=embedding_model,
                                        allow_dangerous_deserialization=True
                                        )  # Load the existing database
        print(f"✅ FAISS database loaded with {len(docs)} existing documents.")
    
    else:
        vector_db = FAISS.from_documents(docs, embedding=embedding_model)
        print("✅ Created a new FAISS database.")   


    # Add new papers to the existing database
    vector_db.add_documents(docs)  # Add new chunks of text to the vector store
    
    # Save the vector database locally
    vector_db.save_local(DB_NAME)
    print(f"✅ FAISS database updated with {len(docs)} chunks!")

    return vector_db

# Function to load FAISS database
def load_faiss_db(user_query="",max_papers=5,generate_new=True):
    return upload_papers_to_faiss(user_query,max_papers,generate_new)

