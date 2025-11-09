# ingest.py
import os
from typing import List
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

from config import (
    ROOT_MATERIALS_DIR,
    SUPPORTED_DOC_EXT,
    SUPPORTED_CODE_EXT,
    PDF_MODEL,
    CODE_MODEL,
    CHROMA_DB_DIR,
    EMBEDDING_MODEL,
    PDF_CHUNK_SIZE,
    PDF_CHUNK_OVERLAP,
    CODE_CHUNK_SIZE,
    CODE_CHUNK_OVERLAP,
    FAST_MODE,
    ALLOWED_EXTENSIONS,
    EXCLUDE_DIR_KEYWORDS,
)

from llm_client import call_local_llm
from parsers import read_pdf, read_py, read_ipynb, read_txt, chunk_text


def scan_files(root: str) -> List[str]:
    """Recursively list only allowed file types under root, skipping unwanted dirs."""
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip dirs like .git, .venv, __pycache__ if you added them in EXCLUDE_DIR_KEYWORDS
        dirnames[:] = [d for d in dirnames if not any(x in d for x in EXCLUDE_DIR_KEYWORDS)]
        for fn in filenames:
            ext = Path(fn).suffix.lower()
            if ext in ALLOWED_EXTENSIONS:
                files.append(os.path.join(dirpath, fn))
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
    print(f"Found {len(all_files)} files before filtering.")

    # here they are already filtered by extension in scan_files, but let's keep it explicit
    filtered_files = [f for f in all_files if Path(f).suffix.lower() in ALLOWED_EXTENSIONS]
    print(f"Found {len(filtered_files)} supported files ({', '.join(ALLOWED_EXTENSIONS)}).")

    # --- connect to chroma ---
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    collection = client.get_or_create_collection(name="ai_library")

    # --- find which files are already indexed ---
    existing_paths = set()
    try:
        # some chroma versions return a flat list, some return list-of-lists
        data = collection.get(include=["metadatas"])
        metas = data.get("metadatas", [])
        for item in metas:
            # item may be dict OR list[dict]
            if isinstance(item, dict):
                if "source_path" in item:
                    existing_paths.add(item["source_path"])
            elif isinstance(item, list):
                for sub in item:
                    if sub and "source_path" in sub:
                        existing_paths.add(sub["source_path"])
    except Exception as e:
        print(f"[WARN] Could not fetch existing metadatas: {e}")

    print(f"Already indexed {len(existing_paths)} files.")

    # we only process files that are not seen before
    new_files = [f for f in filtered_files if f not in existing_paths]
    print(f"Will process {len(new_files)} new files.")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    doc_id_counter = 0

    # REAL WORK HAPPENS HERE 👇
    for fpath in tqdm(new_files):
        ext = Path(fpath).suffix.lower()

        # ---------- DOCUMENTS (pdf, txt) ----------
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

        # ---------- CODE (.py, .ipynb) ----------
        elif ext in SUPPORTED_CODE_EXT:
            if ext == ".py":
                raw_text = read_py(fpath)
            else:
                raw_text = read_ipynb(fpath)

            if not raw_text.strip():
                continue

            chunks = chunk_text(raw_text, CODE_CHUNK_SIZE, CODE_CHUNK_OVERLAP)
            model = CODE_MODEL
            ftype = "code"

        else:
            # should not happen because of filtering, but safe to keep
            continue

        for chunk in chunks:
            if not chunk.strip():
                continue

            # FAST_MODE = don't call LLM, just store chunk
            if FAST_MODE:
                text_to_store = chunk[:1500]
            else:
                text_to_store = summarise_chunk(model, chunk, fpath)
                if not text_to_store.strip():
                    continue

            try:
                embedding = embedding_fn([text_to_store])[0]
            except Exception as e:
                print(f"[WARN] embedding failed for {fpath}: {e}")
                continue

            doc_id = f"doc_{doc_id_counter}"
            doc_id_counter += 1

            try:
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
                print(f"[WARN] failed to add to collection for {fpath}: {e}")
                continue

    print("✅ Ingestion complete.")


if __name__ == "__main__":
    main()
