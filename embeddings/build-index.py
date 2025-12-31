import os
import faiss
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
import numpy as np
import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

DATA_DIR = "../data/web"      # your crawled text files
INDEX_PATH = "index.faiss"
META_PATH = "meta.pkl"
MODEL_NAME = "intfloat/e5-small-v2"

device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.eval()

def embed(texts, batch_size=8):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        with torch.no_grad():
            inputs = tokenizer(
                batch,  
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            outputs = model(**inputs)
            emb = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings)

def chunk_text(text, tokenizer, max_tokens=200, overlap=40):
    sentences = sent_tokenize(text)
    chunks = []

    current_chunk = []
    current_tokens = 0

    for sent in sentences:
        tokens = tokenizer.tokenize(sent)

        # hard truncate single long sentence
        if len(tokens) > max_tokens:
            tokens = tokens[:max_tokens]
            sent = tokenizer.convert_tokens_to_string(tokens)

        sent_token_count = len(tokens)

        if current_tokens + sent_token_count > max_tokens:
            chunks.append(" ".join(current_chunk))

            # overlap by sentences
            overlap_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_chunk
            current_tokens = sum(len(tokenizer.tokenize(s)) for s in current_chunk)

        current_chunk.append(sent)
        current_tokens += sent_token_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

documents = []
sources = []

for file in os.listdir(DATA_DIR):
    if not file.endswith(".txt"):
        continue

    path = os.path.join(DATA_DIR, file)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    for chunk in chunk_text(text, tokenizer):
        if len(chunk) < 200:
            continue
        documents.append(chunk)
        sources.append(chunk)

print(f"Loaded {len(documents)} documents")

embeddings = embed(documents)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(sources, f)

print("Index built successfully")
