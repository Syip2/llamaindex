from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings,
    StorageContext,
    load_index_from_storage
)

from llama_index.llms.openrouter import OpenRouter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.query_engine import RetrieverQueryEngine

import os

load_dotenv()

# =========================
# LLM
# =========================

llm = OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="openai/gpt-4o"
)

Settings.llm = llm


# =========================
# Embedding model
# =========================

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-base-en-v1.5"
)

Settings.embed_model = embed_model


# =========================
# Chunking
# =========================

Settings.text_splitter = SentenceSplitter(
    chunk_size=200,
    chunk_overlap=40
)


# =========================
# Load or Build Index
# =========================

if os.path.exists("./storage"):

    print("Loading existing index...")

    storage_context = StorageContext.from_defaults(
        persist_dir="./storage"
    )

    index = load_index_from_storage(storage_context)

else:

    print("Loading documents...")

    documents = SimpleDirectoryReader(
        input_dir="data",
        recursive=True
    ).load_data()

    print(f"Loaded {len(documents)} documents")

    print("Building index...")

    index = VectorStoreIndex.from_documents(documents)

    index.storage_context.persist("./storage")


# =========================
# Vector Retriever
# =========================

vector_retriever = index.as_retriever(
    similarity_top_k=12
)


# =========================
# BM25 Retriever
# =========================

bm25_retriever = BM25Retriever.from_defaults(
    docstore=index.docstore,
    similarity_top_k=8
)


# =========================
# Hybrid Retriever
# =========================

retriever = QueryFusionRetriever(
    [vector_retriever, bm25_retriever],
    similarity_top_k=8,
    num_queries=1,
    use_async=False
)


# =========================
# Reranker
# =========================

reranker = SentenceTransformerRerank(
    model="BAAI/bge-reranker-base",
    top_n=5
)


# =========================
# Query Engine
# =========================

query_engine = RetrieverQueryEngine(
    retriever=retriever,
    node_postprocessors=[reranker]
)


print("\nChat started. Type 'exit' to quit.\n")


# =========================
# Chat Loop
# =========================

while True:

    question = input("You: ").strip()

    if question.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = query_engine.query(question)

    print("\nAssistant:\n")
    print(response)

    # Show Sources
    if response.source_nodes:

        print("\nSources:\n")

        for node in response.source_nodes:

            meta = node.metadata
            page = meta.get("page_label", "unknown")
            score = node.score

            print(f"Page {page} (score={score:.2f})")