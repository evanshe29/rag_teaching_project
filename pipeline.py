import os
import json
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from unstructured.partition.pdf import partition_pdf

# 路径配置
INPUT_DIR = "data"
OUTPUT_DIR = "extracted"
INDEX_DIR = "index/faiss_index"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ★ 升级后的文本抽取函数（保留结构 + 来源文件名）
def extract_chunks(pdf_path, source_name):
    elements = partition_pdf(filename=pdf_path)
    chunks = []
    for i, el in enumerate(elements):
        text = el.text.strip()
        if text:
            chunks.append({
                "text": text,
                "page": el.metadata.page_number if el.metadata and el.metadata.page_number else -1,
                "order": i,
                "source": source_name  # ★ 新增字段
            })
    return chunks

# 主流程
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(INDEX_DIR, exist_ok=True)

    model = SentenceTransformer(EMBED_MODEL)
    all_chunks = []
    all_texts = []

    for fname in os.listdir(INPUT_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        path = os.path.join(INPUT_DIR, fname)
        chunks = extract_chunks(path, fname)  # ★ 传入原始文件名
        json_path = os.path.join(OUTPUT_DIR, fname.replace(".pdf", ".json"))
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        all_chunks.extend(chunks)
        all_texts.extend([c["text"] for c in chunks])

    embeddings = model.encode(all_texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, "doc.index"))
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)  # ★ 保存结构化块

if __name__ == "__main__":
    main()
