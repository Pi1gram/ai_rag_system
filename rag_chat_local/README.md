# RAG Chat Local

This repository implements a Retrieval-Augmented Generation (RAG) chat application over your own documents (e.g., university lectures). It follows Azure SDK code patterns (wrappers, abstractions) but can run entirely locally (or use Azure free-tier search for small corpora).

## Features
- Ingest text documents (e.g., lecture transcripts) and chunk them.
- Compute embeddings locally via SentenceTransformers + FAISS (or Azure Cognitive Search free tier if enabled).
- Query a local/open-source LLM (via llama-cpp-python) with retrieved chunks for grounded answers.
- Configurable to toggle between Azure services and local implementations.
- Simple chat interface via Gradio.

## Requirements
- Python 3.9+
- A machine with sufficient RAM/CPU; GPU optional for embedding/LLM inference

## Installation
1. Clone the repository:
   ```
   git clone <this_repo_url>
   cd rag_chat_local
   ```
2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Prepare `config.yaml` (see below).
5. Place your lecture text files (e.g., .txt) in a folder and configure ingestion.

## Configuration (`config.yaml`)
```yaml
# Toggle Azure vs local
use_azure_search: false
use_azure_openai: false

# Azure Search settings (only if use_azure_search=true)
azure_search_endpoint: ""
azure_search_api_key: ""
azure_search_index: "lectures"

# Azure OpenAI settings (only if use_azure_openai=true)
azure_openai_endpoint: ""
azure_openai_key: ""
azure_openai_deployment: ""

# Local settings
local_embedding_model: "all-MiniLM-L6-v2"
faiss_index_path: "faiss_index.bin"
chunk_size: 400         # words per chunk
chunk_overlap: 50       # overlap words
local_llm_model_path: "./models/llama3-7b-q4.bin"  # path to quantized model

# Ingestion
data_folder: "./data"  # folder containing lecture text files
```

## Usage
1. Configure `config.yaml`.
2. Prepare lecture text files under `data_folder`.
3. Ingest and build index:
   ```
   python src/app.py --ingest
   ```
4. Run chat UI:
   ```
   python src/app.py
   ```
5. In the browser, interact with the chat interface.