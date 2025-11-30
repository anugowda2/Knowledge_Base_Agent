# evaluate.py
import os
import csv
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from dotenv import load_dotenv

load_dotenv()

STORE_PATH = Path("store.pkl")
TEST_FILE = Path("test_questions.csv")
DEFAULT_MODEL = "all-MiniLM-L6-v2"

def load_store():
    with open(STORE_PATH, "rb") as f:
        return pickle.load(f)

def get_local_model(name):
    return SentenceTransformer(name)

def get_embedding_local(model, text):
    return model.encode([text], convert_to_numpy=True, normalize_embeddings=True)

def retrieve(store, query_emb, top_k=4):
    embs = store["embeddings"]
    sims = (embs @ query_emb.T).squeeze()
    idx_sorted = np.argsort(-sims)
    results = []
    for idx in idx_sorted[:top_k]:
        results.append(store["texts"][idx])
    return results

def score_answer(ans, expected_keywords):
    keywords = expected_keywords.split("|")
    scores = []
    for k in keywords:
        ratio = fuzz.partial_ratio(k.lower(), ans.lower())
        scores.append(ratio)
    return max(scores)

def main():
    store = load_store()
    model_name = store.get("model_name", DEFAULT_MODEL)
    model = get_local_model(model_name)
    rows = []
    with open(TEST_FILE, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            q = r['question']
            expected = r['expected_keywords']
            q_emb = get_embedding_local(model, q)
            contexts = retrieve(store, q_emb, top_k=4)
            answer_text = "\n\n".join(contexts)
            sc = score_answer(answer_text, expected)
            ok = sc >= 60
            rows.append({"question": q, "score": sc, "pass": ok})
            print(f"Q: {q}\nScore: {sc} Pass: {ok}\n{'-'*40}")
    passed = sum(1 for r in rows if r['pass'])
    print(f"\nPassed {passed}/{len(rows)} tests.")

if __name__ == "__main__":
    main()
