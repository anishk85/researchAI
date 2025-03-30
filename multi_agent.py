from langchain.tools import Tool
from langchain_community.llms import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
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

# Summarization function
def summarize_text(title_abstract : list):
    """Summarizes each document individually."""
    
    summaries = []
    
    for doc in title_abstract:
        summary = llm.invoke(f'''Summarize the following research paper. Use web search if needed 
                              and ensure the response is clear, plain text with no unnecessary 
                              symbols, new lines, or formatting characters. Provide links if relevant:
                              \n{doc}
        ''')
        summaries.append(summary.content)
    
    return summaries


def extract_pagecontent(documents):
    """Creates a list of formatted strings, each containing a document's title and abstract."""
    return [f"Title: {doc['title']}\nAbstract: {doc['abstract']}\n" for doc in documents]

# Define Summarization Tool
# summarization_tool = Tool(
#     name="SummarizationTool",
#     func=summarize_text,
#     description="Summarizes content of research papers into a concise and meaningful format to the user."
# )

# Summarization Agent
# summarization_agent = initialize_agent( 
#     tools=[summarization_tool], 
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=ConversationBufferMemory()
# )

# Citation Agent
# citation_agent = initialize_agent(
#     tools=[],
#     llm=llm,
#     agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
#     memory=ConversationBufferMemory()
# )

# Chat Agent
def chat_agent(prompt):
    """Creates a chat agent that can respond to user queries."""
    return llm.invoke(prompt)


# chat_agent = initialize_agent(
#     tools=[],
#     llm=llm,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     memory=ConversationBufferMemory()
# )

# summary = summarization_agent.invoke(extract_pagecontent(search_results))
# print(summary)  
# citations = citation_agent.invoke(summary)
# response = chat_agent.run(f"Summarized research: {summary}\nCitations: {citations}")

# response = llm.invoke(query)
# print("Final Response:", response)


# print("Search Results:",search_results)  # Print the first search result


