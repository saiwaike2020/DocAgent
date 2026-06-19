# DocAgent

## Purpose
    prototype for scanned-PDF → searchable KB → retrieval + LLM answer pipeline.

## Structure: lightweight RAG agent with modules:

    - main.py — CLI entry, arg parsing, orchestrates PDF->MD conversion -> KB -> interactive QA loop.
    - qianwen_pdf_parser.py — multimodal OCR via remote model; converts PDF → MD, save/load MD file.
    - knowedge_base.py — parses MD into pages/chunks, builds BM25 + dense embeddings, persists to ChromaDB.
    - rag_agent.py — retrieves evidence, builds prompt, calls LLM, expects structured JSON.
    - config.py, config.yaml, logging utils.
    - Data & artifacts: sample.pdf, saved Markdown in output, Chroma DB in db.

# Setup
    ## install uv

    ## use uv to setup environment
     - uv python pin 3.10   (specify python version)
     - uv venv              (create a virtual environment)
     - uv sync              (synchronous dependency)

# Run
    ## setup environment variable
        API_KEY=(Your API_KEY for LLM)
         - PS：$env:API_KEY="sk-ws-H.R"
        NOTE: we use LLM to process PDF, it have to support "multimodal model", for example, model name is 'qwen-vl-plus'.
                you have to make sure your API_KEY have right to process image
    ## command
     - python main.py /dir/sample.pdf --rebuild  (use specify PDF file to generate new knowledge base)
     - python main.py /dir/sample.pdf (same to above)
     - python main.py --rebuild (re-generate knowledge from local md file or default PDF file)
     - python main.py
    
    ## arguments
     - specify full path of PDF file, for example, /dir/sample.pdf. It's optional. 
        If it is none, local sample PDF file will be used.
     - --rebuild it will forces application re-read PDF file, clean and re-generate knowledge base

     ## others
      - set path for cache results from parser PDF file, so application could load local results, not request LLM to save tokens
        you should specify 'ocr.md_save_path' in config.yaml with a directory which can be written.