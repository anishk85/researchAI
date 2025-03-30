flowchart TB
    User --> ChatUI
    ChatUI --> QueryEngine
    QueryEngine --> CacheCheck{Redis Cache?}
    CacheCheck -->|Miss| Retriever
    Retriever --> VectorDB[(Weaviate)]
    Retriever --> KeywordSearch[(BM25/Postgres)]
    QueryEngine --> LLM
    LLM -->|RAG| Context[Document Chunks]
    LLM -->|Generate| Response


