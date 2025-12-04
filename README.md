# Knowledge_Base_Agent

Live demo: https://knowledgebaseagent-v3bfwk6ygbpbkbxxvpx4z2.streamlit.app/  
GitHub repo: https://github.com/anugowda2/Knowledge_Base_Agent

---

## Overview

**Knowledge_Base_Agent** is a PDF-based Knowledge Retrieval Agent that ingests one or more PDF documents, converts them into overlapping text chunks, builds local semantic embeddings (HuggingFace sentence-transformers), and serves a Streamlit UI for evidence-based question answering.

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

---

## Project structure

Knowledge_Base_Agent/

├── app.py

├── ingest.py

├── evaluate.py

├── requirements.txt

├── README.md

├── store.pkl # generated after running ingest.py

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

python ingest.py

Run the app

streamlit run app.py

Open http://localhost:8501 and ask questions.

Deploy to Streamlit Community Cloud (steps you can follow)

Commit your project to GitHub:

git add .
git commit -m "Prepare for Streamlit deploy"
git push origin main

Go to https://share.streamlit.io
 and sign in with GitHub.

Note: this repo is already deployed at:
https://knowledgebaseagent-v3bfwk6ygbpbkbxxvpx4z2.streamlit.app/
