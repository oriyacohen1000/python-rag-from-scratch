# Python RAG from Scratch

A framework-free implementation of a **Retrieval-Augmented Generation (RAG)** pipeline built with Python and the OpenAI API.

The project intentionally avoids orchestration frameworks like LangChain or LlamaIndex to demonstrate a direct understanding of the underlying mechanics: document chunking, embedding generation, vector similarity search, and context-constrained answer generation.

---

## How it works

```
knowledge.txt
     │
     ▼
 Chunking (sliding window)
     │
     ▼
 Embeddings (text-embedding-3-small) ◄── stored in memory
     │
     ▼
 User query ──► query embedding
                     │
                     ▼
              Dot product similarity
                     │
                     ▼
             Top-K relevant chunks
                     │
                     ▼
             GPT-4o (context-only prompt)
                     │
                     ▼
                  Answer
```

**Why dot product?** OpenAI embedding vectors are L2-normalized (unit length), so dot product is equivalent to cosine similarity — no extra computation needed.

---

## Engineering highlights

- **No external vector database** — embeddings are stored in a plain Python list and scored with a manual dot product loop
- **Sliding-window chunking** — configurable `chunk_size` and `overlap` preserve semantic continuity across chunk boundaries
- **Embeddings computed once per session** — the corpus is embedded at startup and reused across all queries, avoiding redundant API calls
- **Dynamic K adjustment** — if the model cannot answer from K=1 chunks, the user is prompted to increase K before re-querying
- **Context-constrained generation** — the prompt explicitly instructs the model to answer only from retrieved context, or respond with `"i don't have enough data"`

---

## Modes

### 1. Interactive RAG Chat
Ask questions against `knowledge.txt` in real time. Chunk size, overlap, and K are configurable at startup.

### 2. Batch Q&A Processor
Reads questions from `question.txt`, runs each through the full RAG pipeline, and writes structured answers to `answers.txt`.

---

## Tech stack

| Component | Choice |
|-----------|--------|
| Language | Python 3.8+ |
| LLM | `gpt-4o` |
| Embedding model | `text-embedding-3-small` |
| Similarity metric | Dot product (≡ cosine for normalized vectors) |
| Vector store | In-memory list |
| Secrets | `python-dotenv` |

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/oriyacohen1000/python-rag-from-scratch.git
cd python-rag-from-scratch
pip install -r requirements.txt
```

### 2. Add your API key

```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 3. Run

```bash
python main.py
```

Choose mode 1 for interactive chat or mode 2 for batch processing.

---

## Project structure

```
python-rag-from-scratch/
├── main.py          # Full RAG pipeline, chunking, retrieval, generation, CLI
├── knowledge.txt    # Document used as the knowledge base
├── question.txt     # Input questions for batch mode
├── answers.txt      # Generated output (created at runtime, gitignored)
├── requirements.txt
├── .env.example
└── tests/
    └── test_rag.py  # Unit tests for chunking and retrieval logic
```

---

## Running tests

```bash
pytest tests/ -v
```

Tests cover:
- Sliding-window chunking (exact size, overlap, short text)
- Retrieval ordering by dot product score
- Batch mode calls the full RAG pipeline (not bare GPT)

---

## Tuning tips

| Parameter | Effect | Good starting point |
|-----------|--------|---------------------|
| `chunk_size` | Larger = more context per chunk, less precision | 500–800 |
| `overlap` | Higher = better continuity across boundaries | 10–15% of chunk_size |
| `K` | Higher = more context retrieved, higher cost | Start at 1, increase if needed |
