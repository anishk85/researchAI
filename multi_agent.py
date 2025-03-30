from langchain.tools import Tool
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from transformers import pipeline
from faiss_db import load_faiss_db
from huggingface_hub import login
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import requests

load_dotenv()   

# HF_TOKEN=os.getenv("HF_TOKEN")
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN  # Set the token in the environment variable
# if HF_TOKEN:
#     login(HF_TOKEN)
# else:
#     raise ValueError("Hugging Face token is missing! Set HF_TOKEN in .env")


# Load FAISSx

# Use LangChain's HuggingFacePipeline wrapper
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7)

# print(HF_TOKEN)
# Search relevant docs
# query = "transformers performance were awesome what is blackbox behind it"
# search_results = faiss_search_tool(query)

# Get fiass db init
def get_faiss_db(query, max_papers :int =5, generate_new=True):
    vector_db = load_faiss_db(query,max_papers,generate_new)  # Load the FAISS database
    return vector_db

# Tool for FAISS Search
def faiss_search_tool(query : str ,max_papers : int = 5, k : int = 2,generate_new : bool =True):
    vector_db=get_faiss_db(query, max_papers, generate_new)  # Load the FAISS database
    # Perform similarity search in the FAISS vector store
    lats=vector_db.similarity_search(query, k,score_threshold=0.5)
    print("🔎 Retrieved Papers:", [doc.metadata["title"] for doc in lats])
    return lats  # Adjust 'k' for the number of results
2
# Summarization function
def summarize_text(text):
    """Summarizes the content of research papers."""
    return llm.invoke(f'''Summarize the following text from given papers and also user web search to retrieve more information and then merge it together meaningfully also in response you should not give any type of unrequired symbols or ** or new line characters 
                      or anything give it clear and plain text and you can also provide links if needed from web search :\n{text}
    ''')

def extract_pagecontent(documents):
    """Extracts the page content from documents."""
    return "\n".join([doc.page_content for doc in documents])

# Define Summarization Tool
summarization_tool = Tool(
    name="SummarizationTool",
    func=summarize_text,
    description="Summarizes content of research papers into a concise and meaningful format to the user."
)

# Summarization Agent
summarization_agent = initialize_agent( 
    tools=[summarization_tool], 
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    memory=ConversationBufferMemory()
)

# Citation Agent
# citation_agent = initialize_agent(
#     tools=[],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=ConversationBufferMemory()
# )

# Chat Agent
# chat_agent = initialize_agent(
#     tools=[],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=ConversationBufferMemory()
# )

# summary = summarization_agent.invoke(extract_pagecontent(search_results))
# print(summary)  
# citations = citation_agent.invoke(summary)
# response = chat_agent.run(f"Summarized research: {summary}\nCitations: {citations}")

# response = llm.invoke(query)
# print("Final Response:", response)


# print("Search Results:",search_results)  # Print the first search result


