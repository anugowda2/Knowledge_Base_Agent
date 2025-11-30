# ingest.py
import os
import pickle
from pathlib import Path
from pypdf import PdfReader
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DATA_DIR = Path("docs")
OUT_FILE = Path("store.pkl")
MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, high-quality
CHUNK_SIZE = 900
OVERLAP = 100

def read_pdf(path: Path):
    text = []
    reader = PdfReader(str(path))
    for p in reader.pages:
        try:
            text.append(p.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text)

def read_docs(folder: Path):
    docs = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() == ".pdf":
            txt = read_pdf(p)
        elif p.suffix.lower() in [".txt", ".md"]:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        else:
            continue
        if txt.strip():
            docs.append({"source": p.name, "text": txt})
    return docs

def chunk_text(text, size=CHUNK_SIZE, overlap=OVERLAP):
    text = text.replace("\r\n", "\n")
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # advance with overlap
        start = max(end - overlap, end)
    return chunks

def embed_texts(texts, model):
    # model is a SentenceTransformer instance
    # returns numpy array of shape (len(texts), dim)
    embs = model.encode(texts, batch_size=16, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return embs

def main():
    print("Loading model:", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    print("Reading documents from docs/ ...")
    docs_raw = read_docs(DATA_DIR)
    if not docs_raw:
        raise SystemExit("No docs found in docs/. Copy your PDF(s) there and re-run.")

    all_chunks = []
    meta = []
    for d in docs_raw:
        chunks = chunk_text(d["text"])
        for i, c in enumerate(chunks):
            all_chunks.append(c)
            meta.append({"source": d["source"], "chunk_id": i})
    print(f"Total chunks: {len(all_chunks)}")

    print("Creating embeddings (this may take a minute, downloads the model first time)...")
    embs = embed_texts(all_chunks, model)
    print("Embedding shape:", embs.shape)

    print("Building NearestNeighbors index...")
    nn = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn.fit(embs)

    store = {"meta": meta, "embeddings": embs, "nn": nn, "texts": all_chunks, "model_name": MODEL_NAME}
    with open(OUT_FILE, "wb") as f:
        pickle.dump(store, f)
    print(f"Saved vector store to {OUT_FILE}")

if __name__ == "__main__":
    main()
