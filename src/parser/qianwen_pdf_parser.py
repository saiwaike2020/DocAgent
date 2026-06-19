import pymupdf as fitz
import os
import requests
import base64

from config.config import load_config
from utils.logger import get_logger

class PDFToMarkdown:
    def __init__(self):
        """
        初始化PDF转Markdown转换器
        
        Args:
            api_key: 千问多模态API密钥
        """
        config = load_config("ocr")
        self.base_url = config.get("url")
        self.model = config.get("model")
        self.api_key = config.get("api_key")
        self.md_save_path = config.get("md_save_path")
        self.md_file = config.get("md_file")
        self.logger = get_logger(self.__class__.__name__)

        if not self.md_save_path and not os.path.exists(self.md_save_path):
            self.md_save_path = None

        if not self.api_key:
            raise ValueError("请提供多模态API密钥")
        
    def pdf_to_base64_images(self, pdf_path, dpi=150):
        """
        将PDF文件转换为Base64编码的图片列表
        
        Args:
            pdf_path: PDF文件路径
            dpi: 图片分辨率
            
        Returns:
            Base64编码的图片列表
        """
        # 打开PDF
        doc = fitz.open(pdf_path) 
        base64_images = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # 设置缩放比例
            zoom = dpi / 72.0  # 72是默认DPI
            mat = fitz.Matrix(zoom, zoom)
            
            # 渲染页面为图片
            pix = page.get_pixmap(matrix=mat)
            
            # 直接转换为Base64，不保存到本地
            img_bytes = pix.tobytes("png")
            base64_image = base64.b64encode(img_bytes).decode('utf-8')
            base64_images.append(base64_image)
            
            # 释放内存
            pix = None
            
        doc.close()
        return base64_images
    
    def recognize_page_with_qwen_vl(self, base64_image, page_num, total_pages):
        """
        使用千问多模态识别单页图片内容
        
        Args:
            base64_image: Base64编码的图片
            page_num: 当前页码
            total_pages: 总页数
            
        Returns:
            识别结果
        """
        # 构建提示词
        prompt = f"""
            请将此PDF页面转换为Markdown格式，特别注意以下要求：
            
            1. 表格格式：识别并准确转换所有表格，使用标准Markdown表格语法：
            | 列1 | 列2 | 列3 |
            |-----|-----|-----|
            | 数据1 | 数据2 | 数据3 |
            
            2. 保留原文档的结构，包括标题层级、段落、列表等
            
            3. 文字内容需要完整保留，不要遗漏任何信息
            
            4. 识别扫描页中的文字内容，确保OCR准确性
            
            5. 在文档开头添加页码信息：第{page_num}页/共{total_pages}页
            
            6. 如果是扫描页，请精确OCR识别所有文字内容，特别是表格中的数据
            
            7. 保持原始排版结构，确保表格列对齐
            
            8. 表格标题应使用粗体标记
            
            9. 重要的数字、日期、关键词应保留格式
            
            10. 保持原始文档的语义结构
            """
        
        # 构建请求数据
        data = {
            "model": "qwen-vl-plus",  # 使用视觉语言模型
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "text": prompt
                            },
                            {
                                "image": f"data:image/png;base64,{base64_image}"  # Base64图片数据
                            }
                        ]
                    }
                ]
            },
            "parameters": {
                "max_tokens": 8192
            }
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                self.base_url,# type: ignore
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                # 提取实际内容
                try:
                    content = result['output']['choices'][0]['message']['content']
                    
                    # 提取文本部分
                    text_content = ""
                    for item in content:
                        if 'text' in item:
                            text_content += item['text']
                    
                    return text_content
                except KeyError as e:
                    self.logger.error(f"解析响应失败: {e} \n响应内容: {result}")
                    raise RuntimeError("第{page_num}页解析失败") from e
            else:
                self.logger.error(f"API请求失败，状态码: {response.status_code}, 错误信息: {response.text}")
                raise RuntimeError("第{page_num}页API请求失败")
                
        except Exception as e:
            self.logger.exception(f"请求识别第{page_num}页时出错")
            raise RuntimeError(f"识别第{page_num}页失败") from e
    
    def convert_pdf_to_markdown(self, pdf_path):
        """
        将PDF完整转换为Markdown文档（无本地存储）
        
        Args:
            pdf_path: PDF文件路径
        """
        self.logger.info(f"开始处理PDF: {pdf_path}")
        
        # 转换PDF为Base64图片列表
        self.logger.info("正在将PDF转换为Base64图片...")
        base64_images = self.pdf_to_base64_images(pdf_path)
        total_pages = len(base64_images)
        
        self.logger.info(f"PDF共{total_pages}页，已转换为Base64图片")
        
        # 识别每页内容并生成Markdown
        markdown_content = f"# {os.path.basename(pdf_path)}\n\n"
        markdown_content += f"> 文档总页数: {total_pages}\n\n"
        
        for i, base64_img in enumerate(base64_images):
            page_num = i + 1
            self.logger.info(f"正在识别第{page_num}页/共{total_pages}页...")
            
            page_content = self.recognize_page_with_qwen_vl(
                base64_image=base64_img,
                page_num=page_num,
                total_pages=total_pages
            )
            
            # 添加页分隔和页码信息
            markdown_content += f"\n---\n"
            markdown_content += f"## 第{page_num}页\n\n"
            markdown_content += page_content
            markdown_content += "\n\n"
       
        return markdown_content
    
    def save_md_file(self, content) -> str:
        from pathlib import Path
        if self.md_save_path:
                out_dir = Path(self.md_save_path)
                # make relative paths explicit against cwd so behavior is predictable
                if not out_dir.is_absolute():
                    out_dir = (Path.cwd() / out_dir).resolve()
        else:
                out_dir = Path(__file__).resolve().parents[2] / "data" / "output"

        out_dir.mkdir(parents=True, exist_ok=True)
        import time
        file_name = "result_" + time.strftime("%Y%m%d_%H%M%S") + ".md"
        output_md = os.path.join(out_dir, file_name)

        try:
            with open(output_md, "w", encoding="utf-8") as f:
                f.write(content)
            self.logger.info(f"文件已成功保存至: {output_md}")

            return output_md
        except Exception as e:
            self.logger.error(f"保存文件至'{output_md}'出错：{e}")
            raise IOError(f"无法保存MD文件到 {output_md}") from e
    
    def get_md_file(self, file_path: str) -> str:
        content = ""
        try:
            self.logger.info(f"从文件: {file_path} 读取文件")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read();
            self.logger.info(f"读取的文件内容长度：{len(content)}")
        except Exception as e:
            self.logger.exception(f"读取 MD 文件 {file_path} 失败")
            raise IOError(f"无法读取MD文件 {file_path}") from e
        
        return content

    def get_md_content(self, pdf_path:str=None, rebuild=False) -> str:
        if rebuild:
            md_content = self.convert_pdf_to_markdown(pdf_path)
            if self.md_save_path:
                self.md_file = self.save_md_file(md_content)
        else:
            if self.md_file:
                md_content = self.get_md_file(self.md_file)
            else:
                self.logger.error("未启用重建且未配置 md_file，参见config.yaml中ocr.md_file")

        return md_content

            
