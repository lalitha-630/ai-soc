# AI-SOC Threat Analyzer

AI-powered SOC tool for analyzing security logs and documents.

## Features

* Log anomaly detection (DNS, brute force, exfiltration)
* LLM-based threat explanation
* PDF Q&A using RAG (FAISS + embeddings)
* Streamlit dashboard UI

## Tech Stack

Python, Streamlit, FAISS, Sentence-Transformers, Groq LLM

## Run locally

pip install -r requirements.txt
streamlit run app.py
