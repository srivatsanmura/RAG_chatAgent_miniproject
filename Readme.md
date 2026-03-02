# Project Folder Structure
rag-doc-assistant/
│
├── app/
│   └── streamlit_app.py
│
├── rag/
│   ├── __init__.py
│   ├── config.py
│   ├── logger.py
│   ├── ingest.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectordb.py
│   ├── retriever.py
│   ├── reranker.py
│   ├── generator.py
│   ├── graph.py
│
├── data/
│   ├── raw/
│   ├── vectordb/
│
├── scripts/
│   ├── ingest_docs.py
│   ├── build_index.py
│
├── .env.example
├── requirements.txt
├── README.md