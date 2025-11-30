Knowledge_Base_Agent
AI Agent Development Challenge — PDF-Based Knowledge Retrieval Agent

This project implements a Knowledge Base AI Agent that can ingest PDF documents, build semantic embeddings locally (no paid API required), retrieve relevant document chunks, and answer user queries in a structured, evaluation-ready way.

It follows the AI Agent Development Challenge instructions found in the provided project PDF.

🚀 Features
✔ Local Embeddings (FREE)

Uses HuggingFace Sentence Transformers (all-MiniLM-L6-v2) to generate document embeddings:

No OpenAI quota needed

No billing required

Works offline after model download

Fast and accurate semantic search

✔ PDF Ingestion & Chunking

Automatically reads PDFs from the docs/ folder

Splits large documents into overlapping text chunks

Generates high-quality vector embeddings

Stores them in a local store.pkl vector store

✔ Vector Search (Retrieval)

Uses cosine similarity + scikit-learn NearestNeighbors to retrieve the most relevant chunks for any query.

✔ Two Answer Modes

Local Extractive Answer (Default)

Fully free & offline

Merges top retrieved chunks into a concise extractive answer

LLM Answer (Optional)

If you set OPENAI_API_KEY, the app uses GPT to synthesize a clean, context-aware answer

Optional, not required

✔ Streamlit Web App

Includes a clean UI to:

Enter questions

Display top-k retrieved chunks

Show extractive or LLM-generated answers

Run demo live in class or in interview

✔ Evaluation Script

evaluate.py checks your agent’s correctness against a PDF-based test dataset.

📁 Project Structure
Knowledge_Base_Agent/
│
├── docs/                     # Place your PDFs here
│    └── AI_Agent_Development_Challenge.pdf
│
├── ingest.py                 # Reads PDFs → chunks → embeddings → vector store
├── app.py                    # Streamlit demo (local + optional LLM synthesis)
├── evaluate.py               # Evaluates agent accuracy using test_questions.csv
├── test_questions.csv        # Evaluation questions based on challenge PDF
├── requirements.txt          # Dependencies
├── store.pkl                 # Auto-generated vector store (after ingestion)
└── README.md                 # Project documentation (this file)

🔧 Installation & Setup (Windows CMD)
1️⃣ Create virtual environment
python -m venv venv
venv\Scripts\activate.bat

2️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

3️⃣ Add your PDF(s)

Place your challenge PDF here:

Knowledge_Base_Agent/docs/


Example:
AI_Agent_Development_Challenge_for_merge.pdf

🧠 Run PDF Ingestion (Build Vector Store)
python ingest.py


This will:

Read PDFs

Chunk them

Build local embeddings

Create store.pkl

First run may take a minute (model download).

🌐 Run the Agent App (Streamlit UI)
streamlit run app.py


The app will open in your browser at:

http://localhost:8501

📝 Usage
✔ Type a question

Example demo questions:

What are the agent types listed in the challenge?

What should the final submission include?

What is the purpose of embeddings?

How does the architecture workflow look?

✔ The agent will:

Convert your question into a local embedding

Retrieve top-k matching PDF chunks

Show relevant sources

Generate:

Local extractive answer (default), or

LLM synthesized answer (if OPENAI_API_KEY is set)

🧪 Evaluation

Run the evaluator:

python evaluate.py


It checks:

Retrieval quality

Coverage of expected keywords

Basic scoring threshold

Pass/Fail summary