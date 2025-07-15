
import json
import faiss
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# 路径配置
# main.py 中加入（放在文件顶部）
import os
CURRENT_DIR = os.path.dirname(__file__)
INDEX_DIR = os.path.join(CURRENT_DIR, "..", "index", "faiss_index")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# 加载 FAISS 索引和结构化块
def load_data():
    index_path = os.path.join(INDEX_DIR, "doc.index")
    chunks_path = os.path.join(INDEX_DIR, "chunks.json")

    if not os.path.exists(index_path):
        raise FileNotFoundError(f"索引文件不存在: {index_path}")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError(f"文本块文件不存在: {chunks_path}")

    print("🔍 正在加载索引和教学内容...")
    idx = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return idx, chunks

# 检索相似内容
def retrieve(question, idx, chunks, model, topk=5):
    emb = model.encode([question])
    D, I = idx.search(emb, topk)
    return [chunks[i] for i in I[0]]

# GPT 回答
def answer_with_llm(question, context):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
    prompt = PromptTemplate.from_template(
        "根据以下教学内容回答问题：\n{context}\n\n问题：{question}\n答案："
    )
    # ★ 提取纯文本用于 Prompt 构造
    raw_texts = [chunk["text"] for chunk in context]
    full_prompt = prompt.format(context="\n".join(raw_texts), question=question)
    response = llm.invoke(full_prompt)
    return response.content.strip()

# 主程序
if __name__ == "__main__":
    idx, chunks = load_data()
    embed_model = SentenceTransformer(EMBED_MODEL)
    question = input("请输入问题：")
    context = retrieve(question, idx, chunks, embed_model)

    print("\n--- 检索到的上下文 ---")
    for i, c in enumerate(context):
        print(f"[{i + 1}] ({c['source']} 第 {c['page']} 页, 位置 {c['order']}): {c['text']}")

    print("\n--- GPT 回答 ---")
    print(answer_with_llm(question, context))
