from pymongo import MongoClient
import os
from dotenv import load_dotenv
load_dotenv()

MONGO_URL=os.getenv("MONGO_URL")

def db_setup(MONGO_URL):
    try:
        # Connect to MongoDB (Local or Remote)
        client = MongoClient(MONGO_URL)  # For Local

        # client = MongoClient("mongodb+srv://username:password@cluster.mongodb.net/")  # For Atlas

        db = client["research_db"]
        collection = db["papers"]
        collection.create_index("pdf_url", unique=True)
        return db,collection
    

    except Exception as e:
        print(f"❌ Error connecting to MongoDB: {e}")
        return None, None
    
db, collection = db_setup(MONGO_URL)