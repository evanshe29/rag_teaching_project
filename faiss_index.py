import os
import faiss
import numpy as np
import pickle
from langchain.text_splitter import MarkdownHeaderTextSplitter
from sentence_transformers import SentenceTransformer

output_dir = "output_dir"
index_file = "faiss_index.bin"
meta_file = "docs_and_metas.pkl"

def markdown_slice(output_dir):
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1")])
    docs, metas = [], []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".md"):
                file_path = os.path.join(root, file)
                print(f"📄 正在处理: {file_path}")
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()    
                chunks = splitter.split_text(content)
                for chunk in chunks:
                    text_block = chunk.page_content.strip()
                    if not text_block:
                        continue
                    docs.append(text_block)
                    metas.append({
                        "file": file_path,
                        "header": chunk.metadata.get("Header 1", "")})
    return docs, metas

def embed_texts(texts):
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

def build_faiss_index(embeddings):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    return index

def main():
    docs, metas = markdown_slice(output_dir)
    print(f"📄 已切片 {len(docs)} 段")
    embeddings = embed_texts(docs)
    print(f"✅ 向量维度: {embeddings.shape}")

    index = build_faiss_index(embeddings)
    faiss.write_index(index, index_file)

    with open(meta_file, "wb") as f:
        pickle.dump((docs, metas), f)

    print(f"✅ 索引已保存到 {index_file}, 文档元数据保存到 {meta_file}")

if __name__ == "__main__":
    main()
