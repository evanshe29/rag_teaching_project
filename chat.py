import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import os

index_file = "faiss_index.bin"
meta_file = "docs_and_metas.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_texts(texts):
    return embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

load_dotenv(override=True)
def query_deepseek(prompt, model="deepseek-chat"):
    client = OpenAI(
        api_key=os.getenv('DEEPSEEK_API_KEY'),
        base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是一个教学助手，需要结合文档内容回答用户问题，并标注来源。"},
            {"role": "user", "content": prompt}])
    return response.choices[0].message.content

def main():
    index = faiss.read_index(index_file)
    with open(meta_file, "rb") as f:
        docs, metas = pickle.load(f)
    print("✅ 已加载索引，可以开始问答")

    while True:
        query = input("\n❓ 请输入问题（输入 'exit' 退出）：")
        if query.lower() == "exit":
            break
        query_vec = embed_texts([query])
        D, I = index.search(query_vec, k=3)
        retrieved_docs = [docs[i] for i in I[0]]
        retrieved_metas = [metas[i] for i in I[0]]
        context = "\n\n".join(
            [f"[来源: {m['file']} | {m['header']}]\n{d}" 
             for d, m in zip(retrieved_docs, retrieved_metas)])
        final_prompt = f"以下是与问题相关的文档片段：\n{context}\n\n请基于以上内容回答问题：{query}"
        answer = query_deepseek(final_prompt)
        print(f"\n🐋 DeepSeek: {answer}")

if __name__ == "__main__":
    main()

