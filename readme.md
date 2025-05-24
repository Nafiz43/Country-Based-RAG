# Country Based RAG System

This repository provides a simple Retrieval-Augmented Generation (RAG) pipeline that:
- Ingests PDF documents from India and China,
- Stores vector embeddings using `ChromaDB`,
- Tags documents by country,
- Enables country-specific or combined queries using a local language model (`Ollama`).

## Folder Structure

```
Country-Based-RAG/
├── India/       # Place India-related PDFs here
├── China/       # Place China-related PDFs here
```

## Dependencies

Install all requirements:

```bash
pip install -r requirements.txt
```

## How It Works

- **Ingestion**: Parses PDFs with `pdfplumber`, splits into chunks, embeds using `SentenceTransformer`, and stores in ChromaDB with a country tag.
- **Querying**: Retrieves top-k similar chunks based on a prompt, scoped to either `India`, `China`, or `India+China`, and uses `Ollama` to generate answers.

## Usage

### 1. Ingest PDFs

```bash
python rag_module.py --mode ingest
```

This will index all PDFs under `RAG_System/India` and `RAG_System/China`.

### 2. Query the System

#### Query only India-specific content

```bash
python rag_module.py --mode query --scope India
```

#### Query only China-specific content

```bash
python rag_module.py --mode query --scope China
```

#### Query both India and China (2 entries each)

```bash
python rag_module.py --mode query --scope India+China
```

## 🤖 Model Used

- **Embeddings**: `all-MiniLM-L6-v2` via `sentence-transformers`
- **LLM Inference**: `Ollama` running locally (tested with `mistral`)

## 📦 Notes

- Make sure Ollama is running and the model (e.g., `mistral`) is available.
- You can change `mistral` to another local model in the script if needed.
- Top `k=4` results are returned; for `India+China`, it uses 2 from each.