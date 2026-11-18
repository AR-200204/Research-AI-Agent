# Research AI Agent

An AI-powered research assistant that helps explore academic topics, retrieve relevant research evidence, and generate answers grounded in retrieved sources.

Research AI Agent combines local language models, vector embeddings, and Retrieval-Augmented Generation (RAG) to make academic research easier to explore without depending on paid AI APIs.

## Overview

Finding relevant information across academic papers and research documents can take considerable time. This project aims to simplify that process by bringing research retrieval and AI-powered analysis into one workflow.

Users can ask research questions in natural language. The system searches its vector database for relevant text chunks and provides them to a local language model to generate an evidence-based response.

The project is being developed with a focus on local execution, source traceability, and reducing unsupported AI-generated citations.

## Features

- **Academic research collection** – Gather research sources for a selected topic.
- **Text extraction and chunking** – Prepare source content for semantic retrieval.
- **Vector embeddings** – Convert research text into numerical representations using Ollama.
- **Semantic search** – Retrieve relevant passages based on the meaning of a question.
- **RAG-based question answering** – Generate answers using retrieved research evidence.
- **Citation validation** – Detect and block citations that do not match the retrieved chunks.
- **Local AI processing** – Run language models locally through Ollama.
- **Persistent vector storage** – Store research chunks in ChromaDB.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application |
| Ollama | Local model execution |
| Qwen3 4B Instruct | Answer generation |
| nomic-embed-text | Text embeddings |
| ChromaDB | Vector database |
| Requests | Communication with Ollama API |

## How It Works

The current research workflow follows these steps:

1. Collect research sources related to a selected topic.
2. Extract the available text from each source.
3. Divide the content into smaller overlapping chunks.
4. Generate embeddings for each chunk.
5. Store embeddings and source metadata in ChromaDB.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant research chunks.
8. Pass the retrieved evidence to the local language model.
9. Generate an answer with source and chunk references.
10. Validate citations against the retrieved evidence.

```text
Research Sources
      |
      v
Text Extraction
      |
      v
Text Chunking
      |
      v
Embedding Generation
      |
      v
    ChromaDB
      |
      v
User Research Question
      |
      v
Question Embedding
      |
      v
Semantic Retrieval
      |
      v
Retrieved Evidence
      |
      v
Local LLM (Qwen3)
      |
      v
Citation Validation
      |
      v
Research Answer
```

## Getting Started

### Requirements

- Python 3.10 or newer
- Git
- Ollama
- Sufficient RAM and storage to run the selected local models

### 1. Clone the repository

```bash
git clone https://github.com/AR-200204/Research-AI-Agent.git
cd Research-AI-Agent
```

### 2. Create a virtual environment

Windows:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```cmd
pip install chromadb requests
```

### 4. Install Ollama models

Download and install Ollama from:

https://ollama.com/

Then pull the required models:

```cmd
ollama pull qwen3:4b-instruct
ollama pull nomic-embed-text
```

Make sure Ollama is running before starting the application.

### 5. Prepare the vector database

The RAG application expects an existing ChromaDB collection named `research_chunks`.

The database is created from the research source data using the project's vector database builder.

After preparing the research dataset, run:

```cmd
python build_vector_db.py
```

### 6. Start the research assistant

```cmd
python rag_qa.py
```

You can now enter research questions in the terminal.

Type `exit` to close the application.

## Example

**Research question:**

```text
How is machine learning used to classify astronomical objects?
```

The system retrieves relevant research chunks from ChromaDB and uses the local language model to construct an answer based on that evidence.

The application also checks whether generated citations refer to chunks that were actually retrieved.

## Citation and Evidence Handling

One important challenge with AI-generated research answers is fabricated citations.

Research AI Agent includes a citation validation mechanism that compares generated source references against the chunks retrieved from ChromaDB.

Unsupported citations are replaced with:

```text
[UNVERIFIED CITATION]
```

This check confirms that a citation refers to retrieved evidence. It does not yet guarantee that the cited passage fully supports the associated claim. Claim-level evidence verification remains an area for further development.

## Project Structure

```text
Research-AI-Agent/
│
├── backend/
│   └── app/
│       ├── evidence_extractor.py
│       ├── research_analyzer.py
│       └── search_arxiv.py
│
├── research_data/
│
├── reports/
│
├── build_vector_db.py
├── test_vector_search.py
├── rag_qa.py
├── .gitignore
└── README.md
```

*Note: Some generated data and local database directories are intentionally excluded from Git.*

## Current Development Status

The project is under active development.

| Component | Status |
|---|---|
| Research source collection | Implemented |
| Text extraction | Implemented |
| Evidence extraction | Implemented |
| Research report generation | Implemented |
| Vector database integration | Implemented |
| Semantic search | Working |
| RAG question answering | Working |
| Citation validation | Initial implementation |
| Claim-level evidence verification | Planned |
| FastAPI backend | Planned |
| Web interface | Planned |
| Research workspace | Planned |

## Future Improvements

- Improve academic source discovery and metadata extraction.
- Add claim-level evidence verification.
- Improve citation accuracy and source attribution.
- Support larger research collections.
- Develop a FastAPI backend.
- Build a web-based research interface.
- Add research comparison and research gap analysis.
- Allow users to manage multiple research projects.

## Contributing

Suggestions, bug reports, and contributions are welcome.

If you find an issue or have an idea for improving the project, feel free to open an issue or submit a pull request.

## License

A license has not yet been selected for this project.

---

**Research AI Agent** — Making academic research easier to explore, one question at a time.
