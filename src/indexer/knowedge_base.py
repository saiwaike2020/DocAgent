import os
import re
import jieba
from rank_bm25 import BM25Okapi
from modelscope import snapshot_download
from sentence_transformers import SentenceTransformer

import numpy as np
import chromadb

from config.config import load_config
from utils.logger import get_logger

# ---------- 1. 分页与切分函数（保持原有高效逻辑） ----------
def split_md_by_page(md_text):
    raw_pages = re.split(r'\n\s*---\s*\n', md_text)
    pages = []
    page_pattern = re.compile(r'(?:第\s*(\d+)\s*页|Page\s*(\d+)|\[PAGE\s+(\d+)\])', re.IGNORECASE)
    for raw_page in raw_pages:
        if not raw_page.strip():
            continue
        page_num = None
        match = page_pattern.search(raw_page)
        if match:
            extracted = match.group(1) or match.group(2) or match.group(3)
            page_num = int(extracted)
        pages.append((page_num, raw_page.strip()))
    if not pages:
        pages.append((None, md_text))
    return pages

def parse_markdown_to_chunks(page_text, max_chunk_size=600):
    """
    智能 Markdown 分块函数（支持长表格切分与表头继承）
    max_chunk_size 建议设置在 600-800 之间，以完美适配 M3E 模型的 512 token 限制。
    """
    chunks = []
    current_header = "文档内容："
    current_block = []
    current_block_len = 0
    
    # 状态机：用于记录当前是否在表格中，以及缓存表头
    in_table = False
    table_header_lines = []
    
    lines = page_text.split('\n')
    
    for line in lines:
        stripped_line = line.strip()
        
        # 1. 识别 Markdown 标题，更新当前上下文标题
        header_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if header_match:
            # 遇到新标题，立即结算上一个 block
            if current_block and any(c.strip() for c in current_block):
                chunk_text = f"[{current_header}]\n" + "\n".join(current_block)
                chunks.append(chunk_text.strip())
            
            # 初始化新标题的环境
            current_header = header_match.group(2).strip()
            current_block = []
            current_block_len = 0
            in_table = False
            table_header_lines = []
            continue
            
        # 2. 表格状态机检测
        # 判断标准：行以 '|' 开头且包含 '|' 即可认为是表格行
        is_table_line = stripped_line.startswith('|') and '|' in stripped_line[1:]
        
        if is_table_line:
            if not in_table:
                in_table = True
                table_header_lines = [line] # 第一行通常是字段名
            elif len(table_header_lines) == 1 and '---' in stripped_line:
                table_header_lines.append(line) # 第二行通常是格式分割线 |---|---|
        else:
            # 离开表格区域
            if in_table:
                in_table = False
                table_header_lines = []

        # 3. 将当前行加入缓冲池
        current_block.append(line)
        current_block_len += len(line)

        # 4. 长度溢出时的强制切分与继承逻辑
        if current_block_len >= max_chunk_size:
            # 生成当前的 Chunk
            chunk_text = f"[{current_header}]\n" + "\n".join(current_block)
            chunks.append(chunk_text.strip())
            
            # 清空池子准备下一个 Chunk
            current_block = []
            current_block_len = 0
            
            # 【核心逻辑】：如果此时我们正在把一个长表格切成两半
            # 下一个 Chunk 必须带上当前的标题和表头，否则失去语义
            if in_table and table_header_lines:
                # 标注这是一个续接的表格
                current_block.append(f"*(接上表)*") 
                current_block.extend(table_header_lines)
                # 重新计算起始长度
                current_block_len += sum(len(l) for l in table_header_lines) + len("*(接上表)*")

    # 收尾处理：处理文件末尾还没打包的剩余行
    if current_block and any(c.strip() for c in current_block):
        chunk_text = f"[{current_header}]\n" + "\n".join(current_block)
        chunks.append(chunk_text.strip())
        
    return chunks
# ---------- 2. 持久化知识库 ----------
class PersistentMarkdownKB:
    def __init__(self, md_source=None, persist_directory="../../data/db", rebuild: bool = False):
        """
        Args:
            md_source: Markdown文本或文件路径。如果本地已有数据且rebuild=False，该参数可为None。
            from_file: md_source 是否是文件路径
            persist_directory: 本地持久化数据存储目录（会自动创建）
            rebuild: 是否强制清空本地数据并重新
        """
        config = load_config("knoweldge")
        self.logger = get_logger(self.__class__.__name__)
        self.persist_directory = config.get("db_path", "./chroma")
        self.embeding_model_path = config.get("mode_path")
        # self.embed_model = SentenceTransformer('moka-ai/m3e-base')
        # 定义判断模型是否完整的标志性文件（HuggingFace/ModelScope 模型通常必带 config.json）
        model_config_file = os.path.join(self.embeding_model_path, "config.json")

        
        try:
            # ==========================================
            # 核心改造：真正的本地优先加载逻辑
            # ==========================================
            if os.path.exists(model_config_file):
                self.logger.info(f"[Embedding] 检测到本地已存在模型，跳过网络请求，直接纯离线加载: {self.embeding_model_path}")
                self.embed_model = SentenceTransformer(self.embeding_model_path)
                self.logger.info("[Embedding] 模型完全离线加载成功！")
            
            else:
                self.logger.info(f"[Embedding] 本地路径 {self.embeding_model_path} 未找到模型，开始从 ModelScope 下载...")
                # 触发下载，指定 local_dir 后，文件会直接平铺下载到该目录
                model_dir = snapshot_download(
                    model_id="AI-ModelScope/m3e-base", 
                    local_dir=self.embeding_model_path
                )
                self.logger.info(f"[Embedding] 模型已安全下载并缓存至本地路径: {model_dir}")
                
                # 下载完成后加载模型
                self.embed_model = SentenceTransformer(model_dir)
                self.logger.info("[Embedding] 模型在线下载并加载成功！")
                
        except Exception as e:
            self.logger.error(f"模型加载失败。错误信息: {e}")
            raise RuntimeError(f"ModelScope 下载或本地加载失败，详细查看：") from e
        
        # 这会在你指定的目录下生成名为 chroma.sqlite3 的文件和相关的向量索引文件夹
        self.chroma_client = chromadb.PersistentClient(path=self.persist_directory)
        
        # 获取或创建集合
        self.collection = self.chroma_client.get_or_create_collection(
            name="markdown_hybrid_db",
            metadata={"hnsw:space": "cosine"}
        )
        
        # 检查本地现存的数据条数
        existing_count = self.collection.count()
        
        # 智能化判断：如果本地有数据，且用户没有要求“强制重建”，则直接从本地磁盘加载
        if existing_count > 0 and not rebuild:
            self.logger.info(f"[ChromaDB] 检测到本地目录 '{persist_directory}' 已有 {existing_count} 条向量数据，正在直接读取...")
            
            # 从 ChromaDB 本地拉取完整的文档和元数据
            stored_data = self.collection.get(include=["documents", "metadatas"])
            self.chunks = stored_data["documents"]
            
            # 还原内存中的页码映射
            self.chunk_pages = [
                int(meta["page"]) if meta["page"] != "N/A" else None 
                for meta in stored_data["metadatas"]
            ]
        else:
            # 本地没数据，或者用户明确要求重新构建数据库
            if md_source is None:
                raise ValueError("本地持久化库为空，必须提供 md_source 参数以初始化构建。")
                
            self.logger.info(f"[ChromaDB] 本地无数据或指定了 rebuild=True，开始对文档进行解析并构建持久化库...")
                
            # 1. 拆页与分块
            self.pages = split_md_by_page(md_source)
            self.chunks = []
            self.chunk_pages = []
            for page_num, page_text in self.pages:
                page_chunks = parse_markdown_to_chunks(page_text)
                for c in page_chunks:
                    if len(c.strip()) > 10: 
                        self.chunks.append(c)
                        self.chunk_pages.append(page_num)

            if not self.chunks:
                raise ValueError("未能从提供的输入中解析到任何有效的文本块。")

            # 如果用户指定了强制重建，先将本地旧数据彻底抹去
            if existing_count > 0:
                self.logger.info("[ChromaDB] 正在清理旧的持久化集合...")
                self.chroma_client.delete_collection(name="markdown_hybrid_db")
                self.collection = self.chroma_client.create_collection(
                    name="markdown_hybrid_db",
                    metadata={"hnsw:space": "cosine"}
                )

            # 2. 计算向量并批量写入磁盘
            embeddings = self.embed_model.encode(self.chunks, normalize_embeddings=True)
            metadatas = [
                {"page": str(p) if p is not None else "N/A", "preview": c[:50].replace('\n', ' ')}
                for p, c in zip(self.chunk_pages, self.chunks)
            ]

            self.collection.add(
                ids=[str(i) for i in range(len(self.chunks))],
                embeddings=embeddings.tolist(),
                documents=self.chunks,
                metadatas=metadatas
            )
            self.logger.info(f"[ChromaDB] 数据已成功持久化存储至目录: {self.persist_directory}")

        # 【核心注意点】: 尽管向量存在磁盘上，稀疏检索 BM25 依然属于内存算子，
        # 无论从本地加载还是重新构建，每次启动程序都必须在内存中对 self.chunks 重建一次倒排索引。
        # 可以实现本地缓存方案，但由于用到的lib不兼容python3.10，所以没有实现。
        self.tokenized_corpus = [jieba.lcut(chunk.lower()) for chunk in self.chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    # ---------- 3. 检索流（保持 RRF 混合检索） ----------
    def dense_search(self, query, top_k=10):
        q_vec = self.embed_model.encode([query], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=q_vec, n_results=top_k)
        scores = [1 - d for d in res['distances'][0]] 
        ids = [int(i) for i in res['ids'][0]]
        return list(zip(scores, ids))

    def sparse_search(self, query, top_k=10):
        tokens = jieba.lcut(query.lower())
        scores = self.bm25.get_scores(tokens)
        inds = np.argsort(scores)[::-1][:top_k]
        return list(zip(scores[inds], inds))

    def rrf_fusion(self, dense_res, sparse_res, k=60):
        scores_map = {}
        for rank, (_, idx) in enumerate(dense_res, 1):
            scores_map[idx] = scores_map.get(idx, 0) + 1 / (k + rank)
        for rank, (_, idx) in enumerate(sparse_res, 1):
            scores_map[idx] = scores_map.get(idx, 0) + 1 / (k + rank)
        sorted_ids = sorted(scores_map, key=scores_map.get, reverse=True)
        return sorted_ids, [scores_map[i] for i in sorted_ids]

    def search(self, query, top_k=3):
        dense_res = self.dense_search(query, top_k=len(self.chunks))
        sparse_res = self.sparse_search(query, top_k=len(self.chunks))
        fused_ids, fused_scores = self.rrf_fusion(dense_res, sparse_res)
        results = []
        for i, idx in enumerate(fused_ids[:top_k]):
            results.append({
                "rank": i + 1,
                "score": fused_scores[i],
                "page": self.chunk_pages[idx] if self.chunk_pages[idx] is not None else "未知",
                "content": self.chunks[idx]
            })
        return results