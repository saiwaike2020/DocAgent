import argparse
import json
import os
from pathlib import Path
import re
import sys
SYS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SYS_ROOT / "src"))

# from parser.pdf_parser import SmartPDFTableExtractor
from agent.rag_agent import RAGAgent, RAGResponse
from indexer.knowedge_base import PersistentMarkdownKB
from parser.qianwen_pdf_parser import PDFToMarkdown
from utils.logger import setup_logging, get_logger

from pydantic import ValidationError

def main():
    setup_logging()
    logger = get_logger("main")

    # 1. 只需要一句话初始化并定义唯一参数
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", nargs='?', default=None, help="位置参数，可选")
    
    # 2. 布尔参数：rebuild (使用 --rebuild 开启)
    parser.add_argument(
        "--rebuild", 
        action="store_true", 
        help="是否强制重新构建解析索引/缓存"
    )
    
    args = parser.parse_args()
    pdf_path = args.pdf_path
    rebuild_flag = args.rebuild

    # 1. 校验一：检查后缀是否为 .pdf (使用 lower() 兼容 .PDF 大写情况)
    if pdf_path and not pdf_path.lower().endswith('.pdf'):
        print(f"错误: 输入的文件 '{pdf_path}' 后缀不是 .pdf")
        sys.exit(1)  # 非正常退出程序

    # 2. 校验二：检查该文件在磁盘上是否真实存在
    if pdf_path and not os.path.exists(pdf_path):
        print(f"错误: 找不到文件 '{pdf_path}'，请检查路径是否拼写正确")
        sys.exit(1)

     # 创建转换器实例
    try:
        converter = PDFToMarkdown()
        if not pdf_path and rebuild_flag:
            pdf_path = "./data/raw/sample.pdf"

        md_content = converter.get_md_content(pdf_path, rebuild_flag)
        
        if md_content == "":
            raise ValueError("没有可处理的PDF文件")
        
        indexer = PersistentMarkdownKB(md_content, rebuild=rebuild_flag)

        agent = RAGAgent(indexer)
        while True:
            print("\n" + "="*50)
            question = input("your: ")
            if question.strip() == "q":
                break
            # 执行提问
            try:
                json_str = agent.ask(question, top_k=3)
                match = re.search(r'\{.*\}', json_str, re.DOTALL)
                if not match:
                    raise RuntimeError("未能从大模型返回值中提取到有效的 {} JSON 结构")
                    
                cleaned_str = match.group(0)
                
                # 步骤 B：执行标准的 JSON 反序列化
                response_data = json.loads(cleaned_str)
                
                # 步骤 C：防御双重转义（防止模型返回 "{\"is_answerable\": true}"）
                if isinstance(response_data, str):
                    response_data = json.loads(response_data)
                    
                # 步骤 D：终极防御断言
                # 修复 2：这里必须判断为 dict，因为经过 loads 后它就是纯字典
                if not isinstance(response_data, dict):
                    raise ValueError(f"反序列化失败：期望得到字典(dict)，实际得到 {type(response_data).__name__}")
                
                # 步骤 E：安全解包为 RAGResponse 对象
                validated_res = RAGResponse(**response_data)
                

                # 优雅地输出结构化结果
                print(f"能否从文档中回答: {'是' if validated_res.is_answerable else '否'}")
                print(f"引用页码: {', '.join(validated_res.cited_pages) if validated_res.cited_pages else '无'}")
                print("-"*50)
                print("\n")
                print(f"最终答案:\n{validated_res.answer}")
                print("\n")
                print("="*50) 
            except json.JSONDecodeError as json_e:
                logger.error(f"大模型返回的JSON解析错误: {json_e}")
            except ValidationError as e:
                logger.error(f"字段缺失或类型错误：返回的 JSON 结构不符合 Pydantic 定义。具体错误：\n{e}")
            except RuntimeError as e:
                logger.error(f"运行时错误：{e}")


    except Exception as e:
        logger.error(e)
        sys.exit(1)
    


if __name__ == "__main__":
    sys.exit(main())
