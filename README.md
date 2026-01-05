# Second Brain

A **production-grade AI knowledge assistant** for personal knowledge management. Built with modular **RAG** and **LLMOps** pipelines using **ZenML**, it enables intelligent document ingestion, retrieval, and summarization.

## Key Features
- Document ingestion and intelligent text chunking
- Semantic embeddings with **HuggingFace/OpenAI**
- Hybrid retrieval using **MongoDB Atlas Vector Search**
- Summarization dataset creation and quality evaluation
- Contextual retrieval over personal knowledge bases (e.g., Notion)
- LLM finetuning for summarization using **Unsloth**
- Monitoring and experimentation with **Opik/Comet**
- Accurate query answering, document summarization, and contextual reasoning

## Technologies
- Python, ZenML, HuggingFace, OpenAI APIs
- MongoDB Atlas (Vector Search)
- Opik & Comet for monitoring
- Unsloth for LLM finetuning
- Fastapi, Docker for deployment
- smolagents for Agent development

## Getting Started

1. **Clone the repository**
   ```bash
   git clone https://github.com/Parth19Dhimmar/Second-brain.git
   cd Second-brain

## Project Structure

├── apps/  
│ ├── infrastructure/ # Docker and deployment infrastructure  
│ ├── second-brain-offline/ # Offline ML pipelines for preprocessing, RAG setup, and dataset creation  
│ └── second-brain-online/ # Online inference pipeline - the AI assistant

## Acknowledgements

This project is inspired by the open-source **Second Brain AI Assistant Course** by Decoding ML:  
https://github.com/decodingai-magazine/second-brain-ai-assistant-course
