# Advanced Python RAG: Engineering from Scratch

A lightweight, framework-free implementation of a **Retrieval-Augmented Generation (RAG)** system built with Python and OpenAI APIs.

This project focuses on the core mechanics behind RAG systems: document chunking, embedding generation, semantic retrieval, and context-aware answer generation — without relying on orchestration frameworks such as LangChain or LlamaIndex.

## Overview

This project was built from scratch to demonstrate a practical understanding of the RAG pipeline and the engineering decisions behind it.

Instead of using an external vector database or high-level abstractions, the system uses:
- a custom **sliding-window chunking** pipeline,
- an in-memory **vector store**,
- manual **dot product similarity** for retrieval,
- and direct prompt-based answer generation with OpenAI models.

## Engineering Highlights

- **Custom Retrieval Logic**  
  Implements semantic retrieval manually using embeddings and **dot product similarity**, without external vector database frameworks.

- **Sliding-Window Chunking**  
  Splits the knowledge base into overlapping chunks to improve semantic continuity across adjacent text segments.

- **Embedding Reuse for Efficiency**  
  Document embeddings are generated once during setup and reused across multiple queries in the same session, reducing repeated API calls, latency, and cost.

- **Interactive Retrieval Tuning**  
  The system starts with a small retrieval depth (`K=1`) and allows the user to increase it dynamically when the retrieved context is insufficient.

## Key Features

### 1. Interactive RAG Chat
Ask questions against a local knowledge base using a full RAG pipeline:
1. split the source text into chunks,
2. embed the chunks,
3. retrieve the top-K relevant chunks,
4. generate an answer using only the retrieved context.

### 2. Batch Q&A Processor
Read multiple questions from a local file and generate an output report with structured answers.

### 3. Runtime Hyperparameter Control
Configure important retrieval parameters directly from the terminal:
- `chunk_size`
- `overlap`
- `K-value`

### 4. Smart Retrieval Feedback
If the model cannot answer from the current context, the system suggests increasing the retrieval depth to improve coverage.

## Tech Stack

- **Language:** Python 3.8+
- **LLM:** `gpt-4o`
- **Embedding Model:** `text-embedding-3-small`
- **Environment Management:** `python-dotenv`

## Project Structure

- `main.py` — main application logic, retrieval pipeline, terminal interface, and batch processor
- `knowledge.text` — local knowledge base used by the RAG mode
- `question.text` — input file containing batch questions
- `answers.text` — generated output file for batch answers
- `requirements.txt` — project dependencies

## How It Works

### RAG Flow
1. Load the local knowledge base
2. Split it into overlapping chunks
3. Generate embeddings for all chunks
4. Embed the user query
5. Compute similarity scores between the query and stored chunk vectors
6. Retrieve the top-K most relevant chunks
7. Send the retrieved context to the model for constrained answer generation

### Retrieval Strategy
The system uses **dot product similarity** between the query embedding and each stored chunk embedding to rank relevance.

### Answer Constraint
The generation prompt explicitly instructs the model to answer **only** from the retrieved context.  
If the answer is not present, the model must respond with:

`i don't have enough data`

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME






# Advanced Python RAG: Engineering from Scratch 

A high-performance, framework-free implementation of a **Retrieval-Augmented Generation (RAG)** system.
Built with Python and OpenAI, this project focuses on the core mechanics of vector databases, 
semantic search, and dynamic context optimization.

## The Engineering Approach
Unlike implementations that rely on heavy abstractions like LangChain or LlamaIndex, this project is built from the ground up
to demonstrate a deep understanding of the RAG pipeline:
* **Vector Mathematics:** Manual implementation of **Dot Product similarity** for semantic retrieval without external vector DBs.
* **Data Pipelines:** Custom **sliding-window chunking** logic to preserve semantic continuity across text segments.
* **Cost Efficiency:** Embeddings are generated in optimized batches once per session and reused across multiple queries to minimize API latency and token costs.

##  Key Features
* **Dual Operation Modes:**
    1. **Interactive RAG Chat:** Live conversation with your local data using dynamic retrieval.
    2. **Batch Q&A Processor:** Automated bulk processing that reads questions from a file and generates a structured answer report.
* **Dynamic Hyperparameter Tuning:** Real-time configuration of `chunk_size`, `overlap`, and `K-value` directly through the terminal interface.
* **Proactive Context Optimization (Smart K):** A unique feedback loop that detects insufficient 
    context (K=1) and suggests increasing retrieval depth ($K > 1$) to recover missing information.



##  Tech Stack
* **Language:** Python 3.8+
* **AI Models:** * `gpt-4o` for high-reasoning generation.
    * `text-embedding-3-small` for efficient vector space mapping.
* **Security:** Environment-based API key management via `python-dotenv`.

##  Project Structure
* `main.py`: The core engine featuring the retrieval logic, interactive menu, and batch processor.
* `knowledge.text`: The local knowledge base (Source of Truth).
* `question.text`: Input file for the Batch Q&A mode.
* `requirements.txt`: Minimal dependencies for a lightweight, transparent footprint. 

## 🚀 Getting Started

### 1. Installation
```bash
git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME
pip install -r requirements.txt


### 2. Configuration
Create a .env file in the root directory and add your OpenAI credentials:

use this line of code:

OPENAI_API_KEY=your_actual_api_key_here


###3. Execution
Run the main script and follow the interactive on-screen menu:

Bash:

python main.py


Optimization Tip:
For technical or legal documents, a chunk_size of 800 with an overlap of 100 is recommended.
If the system responds with "i don't have enough data", utilize the built-in prompt to dynamically 
increase the K-value to 3 or 5 to broaden the search horizon.

Developed as a showcase of core AI engineering principles
# python-rag-from-scratch
