# LlamaIndex Hybrid RAG System

This project is a powerful **Retrieval-Augmented Generation (RAG)** system that combines **Vector Search** and **BM25 (Keyword Search)** for enhanced accuracy. It also utilizes a **SentenceTransformer Reranker** to ensure the most relevant information is retrieved.



## Features

* **Hybrid Retrieval**: Merges semantic vector search with keyword-based BM25.
* **Reranking**: Optimizes search results using `bge-reranker-base`.
* **Flexible Data Loading**: Easily ingest documents from your local directory.

## Prerequisites

* **Python**: 3.10+
* **API Key**: [OpenRouter API Key](https://openrouter.ai/)

## Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Syip2/llamaindex.git](https://github.com/Syip2/llamaindex.git)
   cd llamaindex

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt

3. **Configure API Key:**
   Create a .env file in the root directory and add your OpenRouter API key:
   OPENROUTER_API_KEY=your_sk_key_here
