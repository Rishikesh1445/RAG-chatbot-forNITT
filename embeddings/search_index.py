import os
import faiss
import pickle
import torch
import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
from rank_bm25 import BM25Okapi
import google.generativeai as genai

# ================== ENV ==================
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ================== GEMINI ==================
LLM_MODEL = "gemini-2.5-flash"
llm = genai.GenerativeModel(LLM_MODEL)

# ================== EMBEDDING MODEL ==================
MODEL_NAME = "intfloat/e5-small-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model.to(DEVICE)

# ================== LOAD INDEX ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
index = faiss.read_index(os.path.join(BASE_DIR, "index.faiss"))
chunks = pickle.load(
    open(os.path.join(BASE_DIR, "meta.pkl"), "rb")
)
# IMPORTANT:
# chunks MUST have been embedded using: "passage: <text>"
# Otherwise retrieval quality will be bad.

# ================== BM25 ==================
tokenized_chunks = [c.lower().split() for c in chunks]
bm25 = BM25Okapi(tokenized_chunks)

# ================== FUNCTIONS ==================
import torch.nn.functional as F

def embed_query(query: str):
    with torch.no_grad():
        inputs = tokenizer(
            ["query: " + query],
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        outputs = model(**inputs)

        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"].unsqueeze(-1)

        # masked mean pooling
        emb = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)

        # L2 normalize (CRITICAL for E5)
        emb = F.normalize(emb, p=2, dim=1)

        return emb.cpu().numpy().astype("float32")



def hybrid_search(query, k=5):
    # VECTOR SEARCH
    q_emb = embed_query(query)
    vec_scores, vec_ids = index.search(q_emb, 10)

    # BM25 SEARCH
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ids = np.argsort(bm25_scores)[-10:]

    # SCORE MERGE
    scores = {}

    for i, idx in enumerate(vec_ids[0]):
        scores[idx] = scores.get(idx, 0) + (1 / (i + 1))

    for i, idx in enumerate(bm25_ids[::-1]):
        scores[idx] = scores.get(idx, 0) + (1 / (i + 1))

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in ranked[:k]]


def rewrite_query(q):
    prompt = f"""
Convert the query into a short keyword-style search query.
Remove filler words.
Keep only core nouns and entities.
Do NOT answer.

Query: {q}
"""
    return llm.generate_content(prompt).text.strip()


def answer_question(context, question):
    prompt = f"""
Answer ONLY using the context below.
If the answer is not present, say exactly:
Not available in NITT data.

Context:
{context}

Question:
{question}
"""
    return llm.generate_content(prompt).text.strip()
