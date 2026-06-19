from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI

from config.config import load_config
from indexer.knowedge_base import PersistentMarkdownKB
from utils.logger import get_logger

# ==========================================
# 1. 定义大模型输出的 Pydantic 数据结构
# ==========================================
class RAGResponse(BaseModel):
    is_answerable: bool = Field(description="根据上下文是否能回答问题")
    answer: str = Field(description="详细回答内容")
    cited_pages: List[str] = Field(description="引用的源文档页码列表")

# ==========================================
# 2. 构建基于 Pydantic 的 RAG Agent
# ==========================================
class RAGAgent:
    def __init__(self, kb_instance:PersistentMarkdownKB):
        """
        初始化 Agent
        :param kb_instance: 初始化的 PersistentMarkdownKB 实例
        :param api_key: 千问（或对应大模型）的 API Key
        :param model_name: 调用的具体大模型版本
        """
        config = load_config("agent")
        self.kb = kb_instance
        self.model_name = config.get("model")
        self.api_key = config.get("api_key")
        self.base_url = config.get("url")
        self.logger = get_logger(self.__class__.__name__)
        
        # 使用通用的 OpenAI 客户端，配置为阿里云千问的兼容网关
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

    def _build_context_prompt(self, retrieved_chunks: list) -> str:
        """将检索到的 Chunk 组装成清晰的上下文"""
        if not retrieved_chunks:
            return "未检索到任何相关上下文。"
            
        context_parts = []
        for idx, res in enumerate(retrieved_chunks):
            # res 结构来自于我们 kb.search 的返回
            page_info = res.get('page', '未知')
            content = res.get('content', '').strip()
            # 标记明确的来源边界，方便大模型阅读
            context_parts.append(f"--- [来源 {idx + 1} | 页码: {page_info}] ---\n{content}\n")
            
        return "\n".join(context_parts)

    def ask(self, query: str, top_k: int = 3) -> str:
        """处理用户提问并返回结构化响应"""
        
        # 1. 知识库检索
        self.logger.info(f"[Agent] 正在检索知识库寻找 '{query}' 的相关内容...")
        retrieved_chunks = self.kb.search(query, top_k=top_k)
        
        # 2. 组装上下文
        context_str = self._build_context_prompt(retrieved_chunks)
        
        # 3. 设定系统提示词（System Prompt）
        system_prompt = (
            "你是一个专业的财报与文档分析助手。你的任务是根据提供的文档上下文，回答用户的问题。\n"
            "【严格指令】\n"
            "1. 你的回答必须、也只能基于以下提供的上下文。\n"
            "2. 绝对不要使用你的内部知识去编造或猜测答案。\n"
            "3. 如果上下文中的信息不足以回答问题，请将 is_answerable 设为 false，并在 answer 中说明情况。\n"
            "4. 仔细核对财务数据与表格内容，提取准确的数值。\n"
            "5. 如果上下文信息不足以回答问题，必须将answer设置为'信息不足，无法回答' \n"
            "6. 【关键】请务必从提供的上下文中提取你引用信息对应的「页码」，并填入 cited_pages 列表中。例如：如果引用了[来源 1 | 页码: 3]，则列表中需包含 '3'。\n"
            "7. 【关键格式要求】你必须以合法的 JSON 格式返回结果，JSON 必须严格包含以下三个字段：'is_answerable' (布尔值), 'answer' (字符串), 'cited_pages' (字符串列表)。\n\n"
            f"【已知上下文信息】\n{context_str}"
        )

        # 4. 调用大模型，利用 Pydantic 强制结构化输出
        self.logger.info(f"[Agent] 正在调用大模型 ({self.model_name}) 进行思考与回答...")
        try:
            # client.beta.chat.completions.parse 会自动将 Pydantic 模型转换为 JSON Schema
            # 并在返回时自动将 JSON 字符串实例化为 Pydantic 对象
            completion = self.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                # response_format=RAGResponse,  # 核心：直接传入 Pydantic 模型
                temperature=0,  # RAG 场景推荐低温度，保证严谨性
                response_format={"type": "json_object"}
            )
            
            # 返回被完美解析的 Pydantic 对象
            return completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"[Agent] 大模型调用失败: {str(e)}")
            raise RuntimeError("大模型调用失败") from e
