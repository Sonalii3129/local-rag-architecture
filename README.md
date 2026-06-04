# Local RAG-Based Knowledge Retrieval System

A local Retrieval-Augmented Generation (RAG) system that allows users to query documents using natural language. The system uses Python, Ollama, ChromaDB, ReActAgent, and Streamlit to perform document ingestion, vector indexing, semantic retrieval, and contextual response generation.

## Overview

This project demonstrates how local LLMs can be combined with vector databases to create a private document-based question-answering system. Instead of relying only on a model’s general knowledge, the system retrieves relevant context from local documents and uses that context to generate more accurate responses.

## Key Features

* Document ingestion and preprocessing
* Text chunking for retrieval-ready document storage
* Embedding-based vector indexing using ChromaDB
* Semantic search over local documents
* Local LLM response generation using Ollama
* Interactive Streamlit interface
* Modular pipeline structure for ingestion, retrieval, and response generation

## Tech Stack

* Python
* Ollama
* ChromaDB
* Streamlit
* ReActAgent
* Vector embeddings
* Retrieval-Augmented Generation

## Architecture

```text
Documents
   ↓
Document Ingestion
   ↓
Text Chunking
   ↓
Vector Embeddings
   ↓
ChromaDB Vector Store
   ↓
User Query
   ↓
Semantic Retrieval
   ↓
Ollama LLM
   ↓
Contextual Response
```

## Project Structure

```text
local-rag-architecture/
│
├── app.py                  # Streamlit application
├── agent.py                # Agent logic
├── ragcreate.py            # Creates vector store from documents
├── ragrun.py               # Runs the RAG pipeline
├── ragtest.py              # Tests retrieval and response behavior
├── customragcreate.py      # Custom document ingestion workflow
├── data/                   # Source documents
├── custom_data/            # Custom input documents
├── chroma_db/              # ChromaDB vector store
└── README.md
```

## How to Run

Clone the repository:

```bash
git clone https://github.com/Sonalii3129/local-rag-architecture.git
cd local-rag-architecture
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

For Windows:

```bash
venv\Scripts\activate
```

For Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Future Improvements

* Add FastAPI backend endpoints
* Add Docker support for easier deployment
* Add retrieval-quality evaluation metrics
* Improve error handling and fallback responses
* Add screenshots and demo examples
