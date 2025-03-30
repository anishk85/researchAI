from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os
from langchain.docstore.document import Document  # Import the Document class
from setup_mongo import db, collection  # Import the MongoDB collection
import shutil

# Load a Hugging Face embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS database directory
DB_NAME = "research_papers_faiss"

def fetch_chunks_from_mongo(pdf_url:str):
    """Fetches chunks from MongoDB for the given `pdf_url`."""
    paper = collection.find_one({"pdf_url": pdf_url})

    # Check if the document was found
    if paper is None:
        raise ValueError(f"No document found in MongoDB for pdf_url: {pdf_url}")
    
    return [
        Document(
            page_content=chunk, 
        ) for chunk in paper["text_chunks"]
    ]   

def delete_faiss_db():
    """Deletes the FAISS database if it exists."""
    if os.path.exists(DB_NAME):
        shutil.rmtree(DB_NAME)
        print("🗑️ FAISS database deleted.")
    else:
        print("ℹ️ No FAISS database found.")




def upload_chunks(chat_query : str, k : int, generate_new : bool , pdf_url : str):
    # handle the logic when generate new is false
    if not generate_new and os.path.exists(DB_NAME):
        print("✅ Loading existing FAISS database...")
        vector_db = FAISS.load_local(DB_NAME, embeddings=embedding_model, allow_dangerous_deserialization=True)
        return vector_db.similarity_search(chat_query, k=k)  # Retrieve top-k results
    
    #  handle the logic when generate new is true
    if generate_new:
        delete_faiss_db()  # Delete the existing FAISS database if generate_new is True


    # Fetch chunks from MongoDB
    docs = fetch_chunks_from_mongo(pdf_url)

    # Create or update FAISS database
    vector_db = FAISS.from_documents(docs, embedding=embedding_model)
    vector_db.save_local(DB_NAME)  # Save changes
    print(f"✅ FAISS database updated with {len(docs)} chunks.")

    return vector_db.similarity_search(chat_query, k=k)  # Retrieve top-k results


# Function to load FAISS database
def load_faiss_db(chat_query,k,generate_new,pdf_url):
    return upload_chunks(chat_query,k,generate_new,pdf_url)

