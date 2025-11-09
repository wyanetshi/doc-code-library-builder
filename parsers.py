# parsers.py
import os
import json
from typing import List
from PyPDF2 import PdfReader
from PyPDF2.errors import PdfReadError

def read_pdf(path: str) -> str:
    """Read a PDF safely. If it's corrupted or half-synced, return ''."""
    try:
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            texts.append(page.extract_text() or "")
        return "\n".join(texts)
    except (PdfReadError, OSError) as e:
        print(f"[WARN] Skipping unreadable PDF: {path} ({e})")
        return ""
    except Exception as e:
        print(f"[WARN] Error reading PDF {path}: {e}")
        return ""

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Error reading TXT {path}: {e}")
        return ""
        
def read_py(path: str) -> str:
    """Read .py safely."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception as e:
        print(f"[WARN] Error reading PY {path}: {e}")
        return ""

def read_ipynb(path: str) -> str:
    """Read .ipynb safely and join code + markdown."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            nb = json.load(f)
        cells = []
        for cell in nb.get("cells", []):
            if cell.get("cell_type") in ("code", "markdown"):
                cells.append("".join(cell.get("source", [])))
        return "\n\n".join(cells)
    except Exception as e:
        print(f"[WARN] Error reading IPYNB {path}: {e}")
        return ""

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    if not text:
        return chunks
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
