"""
ingest.py — 知识库离线向量化脚本

读取 knowledge.txt，使用本地 BGE 中文 Embedding 模型将文档切块并
构建 FAISS 向量索引，持久化保存到本地 faiss_index/ 目录。
"""

from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".venv" / ".env")

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ══════════════════════════════════════════════════════════════
# 1. 读取知识库文本
# ══════════════════════════════════════════════════════════════

kb_path = Path(__file__).resolve().parent / "knowledge.txt"
with open(kb_path, "r", encoding="utf-8") as f:
    raw_text = f.read()

print(f"[ingest] 读取 knowledge.txt: {len(raw_text)} 字符")

# ══════════════════════════════════════════════════════════════
# 2. 文本切块
# ══════════════════════════════════════════════════════════════

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],
)
chunks = splitter.split_text(raw_text)
print(f"[ingest] 切分为 {len(chunks)} 个文本块")

# ══════════════════════════════════════════════════════════════
# 3. 本地 Embedding 模型 + FAISS 向量库构建
# ══════════════════════════════════════════════════════════════

# BAAI/bge-small-zh-v1.5: 中文优化、轻量级 (384维)、MIT 协议可商用
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

# 构建 FAISS 索引
docs = [Document(page_content=chunk) for chunk in chunks]
vector_store = FAISS.from_documents(docs, embeddings)

# 持久化保存
index_dir = Path(__file__).resolve().parent / "faiss_index"
vector_store.save_local(str(index_dir))

print(f"[ingest] FAISS 向量库已保存到 {index_dir}")
print(f"[ingest] 共 {len(chunks)} 个向量, 维度 384")
print("[ingest] 完成!")
