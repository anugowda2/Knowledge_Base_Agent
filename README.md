# Knowledge_Base_Agent

Live demo: https://knowledgebaseagent-v3bfwk6ygbpbkbxxvpx4z2.streamlit.app/  
GitHub repo: https://github.com/anugowda2/Knowledge_Base_Agent

---

## Overview

**Knowledge_Base_Agent** is a PDF-based Knowledge Retrieval Agent that ingests one or more PDF documents, converts them into overlapping text chunks, builds local semantic embeddings (HuggingFace sentence-transformers), and serves a Streamlit UI for evidence-based question answering.

The agent returns **extractive, document-grounded answers** (it shows the exact chunks used to answer). This version is **fully offline / free** — it does not require the OpenAI API.

---

## Features & limitations

**Features**
- Local embeddings using `sentence-transformers` (`all-MiniLM-L6-v2` by default).
- Fast similarity search via scikit-learn `NearestNeighbors` (cosine similarity).
- Streamlit web UI showing:
  - Top-k retrieved chunks with scores
  - Concise extractive answers (from retrieved text)
  - A short application pitch generator (template-based)
- `ingest.py` to build a `store.pkl` vector store from PDFs in `docs/`.
- Evaluation helper (`evaluate.py`) with `test_questions.csv`.

**Limitations**
- No OCR included: scanned-image PDFs will not yield readable text unless OCR is run first.
- No generative LLM synthesis in this version — answers are extractive only.
- Very large document collections may need FAISS or a vector DB for scalability.
- Very long sentences may be split across chunks; tune `CHUNK_SIZE` / `OVERLAP` if required.

---

## Tech stack & APIs used

- Language: **Python 3.9+**
- UI: **Streamlit**
- Embeddings: **sentence-transformers** (HuggingFace) — `all-MiniLM-L6-v2`
- PDF parsing: **pypdf**
- Vector search: **scikit-learn** `NearestNeighbors`
- Data: **numpy**, **pandas**
- Evaluation: **rapidfuzz**
- (Optional) OpenAI **only** if you later choose to enable LLM synthesis — not required.

---

## Project structure

Knowledge_Base_Agent/
├── app.py
├── ingest.py
├── evaluate.py
├── requirements.txt
├── README.md
├── store.pkl # generated after running ingest.py (commit for faster deploy)
└── docs/
└── AI_Agent_Development_Challenge___for_merge.pdf
---

## Setup & run instructions (local)

1. **Clone**
```bash
git clone https://github.com/anugowda2/Knowledge_Base_Agent.git
cd Knowledge_Base_Agent
Create & activate virtualenv (Windows example)

cmd
python -m venv venv
venv\Scripts\activate.bat
Install dependencies

cmd
python -m pip install --upgrade pip
pip install -r requirements.txt

Build vector store

python ingest.py

This creates store.pkl. (First run downloads the HF model.)

Run the app

streamlit run app.py

Open http://localhost:8501 and ask questions.

Deploy to Streamlit Community Cloud (steps you can follow)

Commit your project to GitHub (include store.pkl to avoid downloading model on the cloud and speed deployments):

git add .
git commit -m "Prepare for Streamlit deploy"
git push origin main


Go to https://share.streamlit.io
 and sign in with GitHub.

Click New app → choose the repo anugowda2/Knowledge_Base_Agent, branch main, and set the main file to app.py. Click Deploy.

Streamlit will build the environment and start your app at a public URL. If you included store.pkl, startup will be fast. If not, the platform will download the HF model during build (slower).

Note: this repo is already deployed at:
https://knowledgebaseagent-v3bfwk6ygbpbkbxxvpx4z2.streamlit.app/

Potential improvements (suggested for future work)

OCR ingestion: integrate pdf2image + pytesseract to support scanned PDFs.

Vector DB / FAISS: use FAISS, Chroma or Pinecone for larger datasets and faster scale.

Highlighting: visually highlight matched tokens inside the snippets for clarity.

Conversational mode: preserve context across turns for multi-turn Q&A.

Local LLM: integrate an open-source LLM (7B–13B) for optional on-device synthesis if you have GPU.

CI / Tests: add unit tests for chunking, retrieval, and evaluation scripts.

Dockerize: containerize the app for reproducible deployment.

