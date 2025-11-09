# ingest.py
import os
from typing import List
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from config import (
    ROOT_MATERIALS_DIR, SUPPORTED_DOC_EXT, SUPPORTED_CODE_EXT,
    PDF_MODEL, CODE_MODEL, CHROMA_DB_DIR, EMBEDDING_MODEL,
    PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP
)
from llm_client import call_local_llm
from parsers import read_pdf, read_py, read_ipynb, chunk_text, read_txt

def scan_files(root: str) -> List[str]:
    """Recursively list all files under root."""
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            files.append(full)
    return files

def summarise_chunk(model: str, chunk: str, file_path: str) -> str:
    """Ask the local LLM for a reusable summary of the given text chunk."""
    prompt = f"""You are creating a reusable AI study library.

File: {file_path}

Summarise this chunk in 3–5 bullet points.
Focus on: key idea, technique, library, task, and where to reuse it.

Text:
{chunk}
"""
    return call_local_llm(model, prompt)

def main():
    print(f"Scanning: {ROOT_MATERIALS_DIR}")
    all_files = scan_files(ROOT_MATERIALS_DIR)

    # setup chroma
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="ai_library")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    doc_id_counter = 0

    for fpath in tqdm(all_files):
        ext = Path(fpath).suffix.lower()   # ← define ext here

        # ----------------------------
        # DOCUMENTS (.pdf, .txt)
        # ----------------------------
        if ext in SUPPORTED_DOC_EXT:
            if ext == ".pdf":
                raw_text = read_pdf(fpath)
            elif ext == ".txt":
                raw_text = read_txt(fpath)
            else:
                continue

            if not raw_text.strip():
                print(f"[WARN] Empty or unreadable doc skipped: {fpath}")
                continue

            chunks = chunk_text(raw_text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
            model = PDF_MODEL
            ftype = "doc"

        # ----------------------------
        # CODE FILES (.py, .ipynb)
        # ----------------------------
        elif ext in SUPPORTED_CODE_EXT:
            if ext == ".py":
                raw_text = read_py(fpath)
            elif ext == ".ipynb":
                raw_text = read_ipynb(fpath)
            else:
                continue

            if not raw_text.strip():
                print(f"[WARN] Empty or unreadable code skipped: {fpath}")
                continue

            chunks = chunk_text(raw_text, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP)
            model = CODE_MODEL
            ftype = "code"

        else:
            # ignore data/other files
            continue

        # ----------------------------
        # Summarise + embed each chunk
        # ----------------------------
        for chunk in chunks:
            if not chunk.strip():
                continue

            try:
                summary = summarise_chunk(model, chunk, fpath)
                if not summary.strip():
                    continue

                doc_id = f"doc_{doc_id_counter}"
                doc_id_counter += 1

                embedding = embedding_fn([summary])[0]

                collection.add(
                    ids=[doc_id],
                    documents=[summary],
                    embeddings=[embedding],
                    metadatas=[{
                        "source_path": fpath,
                        "file_type": ftype
                    }]
                )

            except Exception as e:
                print(f"[WARN] Error summarising or embedding {fpath}: {e}")
                continue

    print("✅ Ingestion complete.")

if __name__ == "__main__":
    main()
