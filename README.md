# NITT RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built over the official NIT Trichy (nitt.edu) website to answer academic and administrative queries using grounded, source-based information.

This project focuses on building a **production-style RAG pipeline** with full website ingestion, semantic search, and LLM-based answering — without fine-tuning.

---

## Features

- Full recursive crawling of the NITT website (no missed internal pages)
- Clean HTML text extraction and preprocessing
- Sentence-aware chunking using NLTK
- Semantic search using embeddings and FAISS
- Retrieval-first query pipeline to prevent hallucinations
- Modular backend architecture (ingestion, indexing, retrieval, API)
- Frontend chat interface for user interaction

---

## Tech Stack

**Backend**
- Python
- FastAPI
- FAISS
- NLTK
- BeautifulSoup

**Frontend**
- React

**LLM**
- API-based LLM (no fine-tuning)

---
