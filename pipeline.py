import os, json
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm

INPUT_DIR = "data"
OUTPUT_DIR = "extracted"
INDEX_DIR = "index/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def extract_chunks(pdf_path):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue
            for line in text.split('\n'):
                line = line.strip()
                if line: chunks.append(line)
    return chunks

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"): continue
        path = os.path.join(INPUT_DIR, fname)
        chunks = extract_chunks(path)
        json_path = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".json"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        all_chunks.extend(chunks)

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(INDEX_DIR, "doc.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
