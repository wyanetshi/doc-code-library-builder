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
    PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP,
    FAST_MODE, MAX_FILES, ALLOWED_EXTENSIONS, EXCLUDE_DIR_KEYWORDS
)

from llm_client import call_local_llm
from parsers import read_pdf, read_py, read_ipynb, read_txt, chunk_text


def scan_files(root: str) -> List[str]:
    """Recursively list only allowed file types under root."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip unwanted directories
        dirnames[:] = [d for d in dirnames if not any(x in d for x in EXCLUDE_DIR_KEYWORDS)]

        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:   # ✅ only process allowed extensions
                full = os.path.join(dirpath, fn)
                files.append(full)
    return files


def summarise_chunk(model: str, chunk: str, file_path: str) -> str:
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
    total = len(all_files)
    print(f"Found {total} files before filtering.")

    if MAX_FILES and MAX_FILES > 0:
        all_files = all_files[:MAX_FILES]
        print(f"Processing only first {len(all_files)} files due to MAX_FILES.")

    # setup chroma
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="ai_library")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    doc_id_counter = 0

    for fpath in tqdm(all_files):
        ext = Path(fpath).suffix.lower()

        # DOCS
        if ext in SUPPORTED_DOC_EXT:
            if ext == ".pdf":
                raw_text = read_pdf(fpath)
            else:  # .txt
                raw_text = read_txt(fpath)

            if not raw_text.strip():
                continue

            chunks = chunk_text(raw_text, PDF_CHUNK_SIZE, PDF_CHUNK_OVERLAP)
            model = PDF_MODEL
            ftype = "doc"

        # CODE
        elif ext in SUPPORTED_CODE_EXT:
            if ext == ".py":
                raw_text = read_py(fpath)
            else:  # .ipynb
                raw_text = read_ipynb(fpath)

            if not raw_text.strip():
                continue

            chunks = chunk_text(raw_text, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP)
            model = CODE_MODEL
            ftype = "code"

        else:
            continue

        for chunk in chunks:
            if not chunk.strip():
                continue

            # 👉 FAST MODE: skip LLM, embed chunk directly
            if FAST_MODE:
                text_to_store = chunk[:1500]  # keep it reasonable
            else:
                text_to_store = summarise_chunk(model, chunk, fpath)
                if not text_to_store.strip():
                    continue

            try:
                embedding = embedding_fn([text_to_store])[0]

                doc_id = f"doc_{doc_id_counter}"
                doc_id_counter += 1

                collection.add(
                    ids=[doc_id],
                    documents=[text_to_store],
                    embeddings=[embedding],
                    metadatas=[{
                        "source_path": fpath,
                        "file_type": ftype
                    }]
                )
            except Exception as e:
                print(f"[WARN] Error on {fpath}: {e}")
                continue

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
