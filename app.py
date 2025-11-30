# app.py
import os
import pickle
from pathlib import Path
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Optional OpenAI: only used if OPENAI_API_KEY is present (you can ignore)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

load_dotenv()

STORE_PATH = Path("store.pkl")
DEFAULT_MODEL = "all-MiniLM-L6-v2"
CHAT_MODEL = "gpt-3.5-turbo"  # only used if you set OPENAI_API_KEY

st.set_page_config(page_title="Knowledge Base Agent (Local Embeddings)", layout="centered")
st.title("Knowledge_Base_Agent — Local Embeddings")
st.write("This agent uses HuggingFace sentence-transformers for embeddings (no OpenAI usage required).")

def load_store():
    if not STORE_PATH.exists():
        st.error("store.pkl not found — run ingest.py first.")
        st.stop()
    with open(STORE_PATH, "rb") as f:
        store = pickle.load(f)
    return store

def load_model(name):
    return SentenceTransformer(name)

def get_query_embedding_local(model, query):
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec

def retrieve(store, query_emb, top_k=3):
    # use cosine similarity directly (embeddings are normalized)
    embs = store["embeddings"]
    sims = (embs @ query_emb.T).squeeze()  # dot product = cosine when normalized
    idx_sorted = np.argsort(-sims)
    results = []
    for idx in idx_sorted[:top_k]:
        results.append({"text": store["texts"][idx], "meta": store["meta"][idx], "score": float(sims[idx])})
    return results

# Optional OpenAI client
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if OPENAI_API_KEY and OPENAI_AVAILABLE:
    openai_client = OpenAI()
else:
    openai_client = None

store = load_store()
model_name = store.get("model_name", DEFAULT_MODEL)
st.info(f"Using local embedding model: **{model_name}**")

# lazy-load model
@st.cache_resource
def get_model(name):
    return SentenceTransformer(name)

model = get_model(model_name)

query = st.text_input("Enter your question about the project PDF:")
top_k = st.slider("Top-k retrieved chunks", 1, 6, 3)

if st.button("Get Answer"):
    if not query.strip():
        st.warning("Type a question.")
    else:
        with st.spinner("Computing embedding and retrieving relevant chunks..."):
            q_emb = get_query_embedding_local(model, query)  # shape (1, dim)
            contexts = retrieve(store, q_emb, top_k=top_k)
        st.subheader("Retrieved contexts")
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**{i}. {c['meta']['source']}** (score: {c['score']:.3f})")
            st.write(c['text'][:800] + ("..." if len(c['text'])>800 else ""))
            st.markdown("---")

        # If OpenAI key present & client available, ask LLM to synthesize.
        if openai_client is not None and OPENAI_API_KEY:
            st.info("OPENAI_API_KEY detected — using OpenAI to synthesize final answer.")
            # build prompt
            context_text = "\n\n".join([f"[{c['meta']['source']}] {c['text']}" for c in contexts])
            system_prompt = ("You are an assistant that MUST answer using ONLY the provided CONTEXT. "
                             "If the answer is not present, state: 'I don't know based on the provided documents.'")
            user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION: {query}\n\nAnswer concisely using only the context and cite sources by filename."
            try:
                resp = openai_client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=[{"role":"system","content":system_prompt},{"role":"user","content":user_prompt}],
                    max_tokens=512,
                    temperature=0.0,
                )
                answer = resp.choices[0].message.content
                st.subheader("Answer (synthesized by OpenAI)")
                st.write(answer)
            except Exception as e:
                st.error("OpenAI generation failed: " + str(e))
                st.subheader("Answer (extractive fallback)")
                st.write("\n\n".join([c['text'] for c in contexts]))
        else:
            st.subheader("Answer (extractive — local)")
            # Simple local summarization: provide the concatenated top contexts as the answer,
            # but trim to a reasonable length (2000 chars).
            combined = "\n\n".join([f"[{c['meta']['source']}] {c['text']}" for c in contexts])
            # If too long, truncate to first 2000 chars and add note.
            if len(combined) > 2000:
                combined = combined[:2000].rsplit("\n",1)[0] + "\n\n...(truncated)"
            st.write(combined)
            st.caption("No OpenAI key detected — returning extractive answer composed from retrieved documents.")

# show small help
with st.expander("Demo questions (copy/paste)"):
    st.write("""
- What are the agent types listed in the challenge?
- What must be included in the submission?
- What is the role of embeddings?
- How should the agent avoid hallucination?
""")
