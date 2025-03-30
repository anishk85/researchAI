from flask import Flask, jsonify, request
from langchain.schema import Document
from multi_agent import faiss_search_tool,summarize_text, extract_pagecontent
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app, supports_credentials=False, origins="*")  # Allow all origins
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route("/",methods=["GET"])
@cross_origin()
def home():
    return "Welcome to the Research Paper Search API!"

@app.route('/agent/fetch_docs',methods=['POST'])
@cross_origin()
def fetch_documents():
    data=request.get_json()
    query = data.get("query","")
    k = data.get("k", 2)
    max_papers = data.get("max_papers", 10)  # Default to 5 if not provided
    if not query or not k or not max_papers:
        return jsonify({"error":"query and k parameter is required"}), 400
    # if (k>5 or k<1) or (max_papers>10 or max_papers<1):
    #     return jsonify({"error":"no of papers must be between 1 and 5"}), 400
    # if max_papers < k :
    #     return jsonify({"error":"k must be less than or equal to max_papers"}), 400
    
    search_results=faiss_search_tool(query,max_papers,k,True)
    summary_of_search_results=summarize_text(extract_pagecontent(search_results))
    # Format the response
    response_data = []
    for doc in search_results:
        response_data.append({
            "title": doc.metadata.get("title", "Unknown Title"),  # Extract title if available
            "abstract": doc.metadata.get("abstract", "N/A"),  # Extract abstract if available
            "page_content": doc.page_content,  # Extracted content from the document
            "full_text": doc.metadata.get("full_text",""),  # Assuming `page_content` is the full text
            "pdf_url": doc.metadata.get("pdf_url", "N/A"),  # Extract PDF URL if available
            "summary": summary_of_search_results.content       # Summarized content
        })        

    return jsonify(response_data), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000,debug=True)