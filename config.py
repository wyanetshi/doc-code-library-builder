# config.py
from dotenv import load_dotenv
import os

load_dotenv()  # reads .env file

# ---- PATHS ----
# root folder where your current "by-topic" materials live
ROOT_MATERIALS_DIR = os.getenv("AI_LIB_PATH", "./AI Product Manager") # <-- change the root path in .env

# where we store processed stuff (summaries, vector db, cache)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- LLM / LOCAL MODELS ----
# your local LLM endpoint (Ollama default)
LLM_BASE_URL = "http://localhost:11434/api/generate"

# which model to use for what
PDF_MODEL = "qwen2:7b"
CODE_MODEL = "deepseek-r1:7b"
CHAT_MODEL = "qwen2:7b"  # used in Streamlit to formulate answers

# ---- VECTOR / EMBEDDINGS ----
# we'll use chromadb locally
CHROMA_DB_DIR = os.path.join(OUTPUT_DIR, "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ---- CHUNKING ----
PDF_CHUNK_SIZE = 1200
PDF_CHUNK_OVERLAP = 200

CODE_CHUNK_SIZE = 400
CODE_CHUNK_OVERLAP = 50

# ---- MISC ----
SUPPORTED_DOC_EXT = [".pdf", ".txt"]
SUPPORTED_CODE_EXT = [".py", ".ipynb"]
