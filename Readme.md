## 1. Overview

This project implements a **Developer Documentation Support Assistant**, which is a conversational Assistant using **Retrieval-Augmented Generation (RAG)** to answer questions about LangGraph and LangChain frameworks.

The assistant retrieves information exclusively from provided official documentation and generates grounded responses while preventing hallucinations. It supports conversational follow-ups and refuses to answer when relevant information is unavailable.

This system demonstrates a production-grade RAG pipeline with retrieval, reranking, compression, conversational memory, and hallucination control.

---
# 2. Domain Selection

**Chosen domain:** Technology – Open-source documentation and API manuals

**Specific focus:** LangGraph and LangChain developer documentation

**Justification:**
This domain provides:
- precise, structured technical documentation,
- ground truth for evaluation,
- well-defined terminology suitable for semantic retrieval,
- ideal content for demonstrating reranking effectiveness,  
- clear scenarios for hallucination prevention.

---

# 3. Project Objectives

The objective of this project is to:
1. Build an ingestion process to collect external data, create chunks and store as vector embeddings (along with metadata) into a vector DB.  (in this project, the vectorDB is **ChromaDB (local)**)
2. Build a retrieval process to fetch relevant documents from the vectorDB to do semantic search based on the input query
3. Build a UI interface as a primary conversational assistant.  This interface is built to:
	* support conversation with a user
	* invoke retrieval process,
	* format and invoke LLM with input
	* track conversational history
	* display metrics data visualization

### Completion criteria:

The assistant will:
	1. Answer questions strictly using provided documentation
	2. Support follow-up questions using conversation history
	3. Prevent hallucination by refusing unsupported queries
	4. Demonstrate a complete RAG pipeline including:
	    - document ingestion
	    - chunking
	    - embedding
	    - vector retrieval
	    - reranking
	    - compression
	    - grounded answer generation
	    - metrics visualization

----

# 4. Data Sources

The project will use publicly available official documentation.  After attempting several ways to extract official LangChain documentation by web scraping and manual methods, I have chosen to use the LangChain's github documentation repository for this purpose.

Source: https://github.com/langchain-ai/docs

This source provides documentation for LangChain, LangGraph, DeepAgents and Integrations.
All the documentation are in mdx format.

As part of this project, there were **2175** mdx documents in this repository and all of these files were processed and ingested.

---
# 5. System Architecture

High level System Architecture is as follows:

![[RAG Architecture.png]]
The system is composed of three layers of implementation:
* **Document Ingestion Layer**
* **Query processing, Retrieval and answer generation Layer**
* **User Interface Layer**

Query processing, Retrieval and answer generation flow is implemented as follows:
```
User Query
    ↓
Query Rewrite (make it standalone)
    ↓
Vector Database Retrieval - MMR Search (Top 20 chunks)
    ↓
Reranker Model (Select Top 5 chunks)
    ↓
Compression (Lightweight implementation)
    ↓
Conversation Context Integration
    ↓
LLM Answer Generation
    ↓
Response grounded in retrieved documents
```

# 6. Project Repo structure and Setup
## 6A. Project Structure
```
Project Structure

rag_chatAgent_miniproject/
 ├── app/
 │    └── streamlit_app.py
 ├── rag/
 │    └── agentMemory.py
 │    └── chunking.py
 │    └── compressor.py
 │    └── config.py
 │    └── embeddings.py
 │    └── extract_docs.py
 │    └── generator.py
 │    └── graph.py
 │    └── logger.py
 │    └── models.py
 │    └── reranker.py
 │    └── retriever.py
 │    └── rewriter.py
 │    └── utils.py
 │    └── vectordb.py
 ├── scripts/
 │    └── build_index.py
 │    └── ingest_docs.py
 │    └── urllist.py
 └── README.md
```

## 6B. Setup Instructions

1. Clone the git repository:
``` git clone https://github.com/langchain-ai/docs```

2. Set SOURCE_PATH in the config.py to the location the repository above.
3. Run ingest_docs.py to build Vector DB
4. run Streamlit interface with  
```python -m streamlit run app/streamlit_app.py --server.port 8501```
# 7. Document Ingestion Pipeline

## Step 1: Document Acquisition

The LangChain documentation git repository - https://github.com/langchain-ai/docs is cloned locally to get access to all the markdown files easily

---
## Step 2: Document Loading

Use UnstructuredMarkdownLoader from LangChain's specialized document loaders to read each file into LangChain's document objects.

Next, use MarkdownHeaderTextSplitter to split the documents into organized chunks.

---
## Step 3: Document Chunking

Recursive chunking (**RecursiveCharacterTextSplitter**) will be used to preserve semantic structure.

Configuration:

```id="chunk1"
chunk_size = 500 tokens
chunk_overlap = 100 tokens
```

This ensures:

- semantic completeness
- contextual continuity
- improved retrieval accuracy

Each chunk includes metadata.

---
## Step 4: Embedding Generation


Each chunk is converted into a vector embedding.

Embedding model:

```id="embed1"
all-MiniLM-L6-v2
```

Output:

```id="embed2"
vector dimension: 384
```

The `all-MiniLM-L6-v2` model produces embeddings with **384 dimensions**.
This model is relatively smaller than ```text-embedding-3-small``` of OpenAI which provides 1536 dimensions.  However, this model ran very fast on 2175 documents using local cpu hardware at no cost.

OpenAI Embeddings could not be used because it had exhausted $0.25 quickly while processing through the documents.

---
## Step 5: Vector Storage

Vector database:

```id="vectordb"
ChromaDB
```

ChromaDB enables mmr (**Maximal Marginal Relevance**) search

Stored data:

```id="vectordata"
embedding vector
chunk text
metadata
```

---
# 8. Retrieval Pipeline
When user submits a query:

## Step 1: Query Rewrite

In order to avoid "Context bloat" with growing chat history and eventually impacting vector search as well as LLM responses, it is important to perform **Query Condensation** (or **Query Reformulation**) to generate a simplified standalone query.

###  Sliding Window with Summary
The program here implements ```Agent Memory``` class to handle a simple implementation of conversational memory management.
#### <u>The Logic:</u>

1. **The Window:** We keep the last **$N$** (usually 2 or 3) messages in full text. This preserves immediate context (like "Explain it _simply_").
2. **The Summary:** Everything older than $N$ messages is passed through a "Summarizer" and stored as a single paragraph.
3. **The Purge:** Very old or redundant messages are dropped entirely.

Now, when it is time to rewrite a user's query, we only send:  
**```Rolling Summary + Last 2 Messages + New Query```** to rewrite_llm to create a **standalone query**.

## Step 2: Query Embedding

Once the query is rewritten (or not), the query is converted into embedding vector.
```id="queryembed"
query → embedding vector
```

---
## Step 3: Vector Search : Maximal Marginal Relevance Search

Retrieve top candidates using the following parameters:

```id="retrieve"
top_k = 20             # Final docs returned
fetch_k = 50           # Initial pool for diversity filtering
lambda_mult = 0.5      # Diversity vs relevance balance
```

This maximizes recall.

---

# 9. Reranking

CrossEncoder model - ```all-MiniLM-L6-v2``` is used to perform ranking of retrieved docs against the query, with Sigmoid function as the activation function.

Top-5 documents based on reranked scores are sent to the next step.

# 10. Compression

Here _lightweight token-based compression (not LLM compression)_ is applied to reduce context size deterministically and cheaply — without invoking another LLM.

### Why Compression Is Needed

After reranking we may have:

```text
5 chunks × 500 tokens = 2500 tokens
```

But:
- LLM context window is finite.
- Smaller context = lower cost.
- Smaller context = better signal-to-noise ratio.

The goal is:

* Preserve relevance, remove redundancy, minimize tokens.

#### Types of Lightweight Compression
Several lightweight compression techniques were evaluated, such as:
	1. **Hard Token Clipping**
	2. **<font color=green>Query-Aware Sentence Filtering</font>**
	3. **Reranker-Score-Based Trimming**

Query-Aware Sentence Filtering strategy is applied in this project.  This indeed has certain limitation, such as code samples in the technical documents are split and clipped.  However, this approach is implemented as a prototype of what is possible in production scenarios.

# 11. Conversation Memory

Conversation history is stored and included in subsequent queries.

There are two simultaneous strategies applied for Conversation history management:
1. Entire conversation history is maintained as-is to enable Streamlit UI to display all the history to resemble that of a chat
2. Advanced memory management is done to simplify/reduce cost on LLM calls on rewriting queries as well as answer generation stage

Example:

```id="conv1"
User: What is LangGraph state?
Assistant: [answer]

User: Can state persist across executions?
```

The system combines:

```id="conv2"
current query + previous conversation
```

This enables contextual understanding.
# 12. Answer Generation (LLM call)

The LLM receives:

```id="geninput"
User query
Conversation history
Top retrieved chunks
```

The LLM generates grounded answers using only retrieved context.

$Note:$, with the  memory management (described earlier) implementation, only the most recent conversations (3 conversations) are passed as chat history to LLM.

# 13. Hallucination Prevention

The system prevents hallucination using relevance thresholds and grounded prompting.

If relevance score below threshold:

```id="refusal"
"I cannot find relevant information in the provided documents."
```

This ensures factual accuracy.

# 14. UI Layer

Streamlit is used to builld the UI layer.  The UI layer provides the following features:
1. Capturing user query and displaying the conversational history
2. Retaining all the chat history in memory
3. Displaying the visualization of reranking scores and chunks
4. Triggering the LangGraph workflow for executing the RAG pipeline
5. Engages the Agent Memory management module


# 15. LangGraph as orchestration Layer

This project uses LangGraph to orchestrate the following flow:
```
Query Rewrite
  ↓
Retrieval (mmr)
  ↓
Reranker
  ↓
Compression
  ↓
Answer Generation
```
# 16. Example Use Cases

Example 1:

```id="ex1"
User: What is LangGraph?
Assistant: [retrieved grounded answer]
```

Example 2 (follow-up):

```id="ex2"
User: How does it manage state?
Assistant: [uses conversation + retrieved context]
```

Example 3 (refusal):

```id="ex3"
User: Does LangGraph support blockchain?
Assistant: I cannot find relevant information in the provided documents.
```

# 17. Technologies Used

| Component       | Technology                     |
| --------------- | ------------------------------ |
| Document loader | LangChain                      |
| Chunking        | RecursiveCharacterTextSplitter |
| Embedding model | all-MiniLM-L6-v2               |
| Vector database | ChromaDB                       |
| Reranker        | cross-encoder                  |
| LLM             | gpt-4o-mini                    |
| Framework       | LangChain / LangGraph          |
| Language        | Python                         |

# 18. Future Improvements

This project leads me to take forward a few future improvement ideas:

```
Future Improvements

• Hybrid retrieval (BM25 + embeddings)
• Better code-aware chunking
• Advanced hallucination prevention controls
• Add automatic evaluation pipeline
• Experiment with higher dimensional embeddings for vector DB
```