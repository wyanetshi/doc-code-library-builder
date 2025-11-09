# 🧠 AI Library Builder & Search

This project converts a folder of PDFs, txts, Python scripts, and notebooks into a **local searchable knowledge base**.  
It automatically scans, summarises, and indexes your files using your local large language models (LLMs), and provides a **Streamlit web interface** for searching and exploring them.

---

## 🚀 Overview

The pipeline consists of:

1. **Scanner** — walks through your folders and finds all supported files.  
2. **Parser** — reads and cleans text from PDFs, `.py`, and `.ipynb`.  
3. **Summariser** — uses your local LLMs (Qwen2 & DeepSeek) to create short, reusable summaries.  
4. **Vector Index** — embeds those summaries into a Chroma database for fast semantic search.  
5. **Streamlit App** — provides a chat-like interface to search and ask questions about your materials.

---

## 📁 Folder structure
BuildLibrary/
├── app.py # Streamlit search interface
├── config.py # All settings live here
├── ingest.py # Scans, summarises, and builds the vector index
├── llm_client.py # Local LLM API helper (talks to Ollama)
├── parsers.py # PDF / code / notebook parsers
├── requirements.txt # Python dependencies
└── README.md # This documentation

Output files are created in `./output/chroma_db`.

---

## 🧩 Supported file types

| Type      | Extension       | Model used      |
|-----------|-----------------|-------------------|
| Documents | `.pdf`, `.txt`  | `qwen2:7b`        |
| Code      | `.py`, `.ipynb` | `deepseek-r1:7b`  |
| Other data | any other | ignored (future support planned) |

---

## ⚙️ 1. Prerequisites

### 🖥️ System
- macOS, Linux, or Windows
- Python 3.10 or higher
- At least 16 GB RAM recommended
- ~10 GB free disk space for models and indexes

### 🧠 Local LLM runtime
We use **[Ollama](https://ollama.ai)** for local model serving.

#### Install Ollama
**Mac:**
```bash
brew install ollama
```
Pull models (only once)
Open Terminal (or Command Prompt on Windows):
```bash
ollama pull qwen2:7b
ollama pull deepseek-r1:7b
```

## 🪄 2. Project setup

### Edit config.py

Open the file and update the path of your learning materials:
ROOT_MATERIALS_DIR = "/Users/yourname/Documents/AI-materials"

## 🐍 3. Python environment setup

### Step 1 – Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # Mac / Linux
# OR
.\.venv\Scripts\activate         # Windows PowerShell
```
### Step 2 – Install dependencies

pip install -r requirements.txt

## 🐍 4. Python environment setup

Your summariser and search assistant both rely on a local model endpoint.
You must start Ollama before running ingest.py or app.py.

### Step 1 – Open a new Terminal window

Keep your virtual environment terminal open; open another new one.

### Step 2 – Run Ollama server

```bash
ollama serve &
```
You should see:
Listening on 127.0.0.1:11434 ...

## 🧱 5. Build your AI library (ingestion)

From your main project terminal (with .venv active):

```bash
python ingest.py
```

This process will:
- Walk through your materials folder.
- Read and split files into chunks.
- Summarise each chunk using the right LLM:
  - PDFs, txt → qwen2:7b
  - Code → deepseek-r1:7b
- Embed those summaries into a local Chroma vector database (./output/chroma_db).

💡 It may take several minutes the first time depending on the number of files.

Once complete, you’ll see Ingestion complete.

## 🔍 6. Start the Streamlit search app

```bash
streamlit run app.py
```

This launches a local web server at something like:
http://localhost:8501

Open it in your browser.
Use the interface
- Type a query:
    transformer fine-tuning, langchain PDF loader, pandas groupby time, etc.
- The app searches your indexed summaries.
- It displays:
  - Top relevant notes (with their source file paths)
  - An optional LLM-generated answer (using qwen2:7b)

## ♻️ 7. Updating the library

When you add new PDFs or code files to your materials folder:
python ingest.py

This re-scans and updates the database.

## ⚙️ 8. Changing models or parameters

All settings live in config.py:
- ROOT_MATERIALS_DIR → your learning materials path (change root folder in .env)
- PDF_MODEL → model for PDFs and txt
- CODE_MODEL → model for code
- CHAT_MODEL → model for Streamlit answers
- Chunk sizes, overlap, and embedding model
No other files need editing.

## 🧼 10. Clean-up / restart

To reset everything:
```bash
rm -rf output/chroma_db
python ingest.py
```

To stop Ollama:
```bash
pkill ollama`
```

## ✅ Summary of commands (quick reference)

### One-time setup

brew install ollama
ollama pull qwen2:7b
ollama pull deepseek-r1:7b

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### Every session

ollama serve &          # start local model server
python ingest.py        # build / refresh the index
streamlit run app.py    # launch the search UI

## 🪄 Optional convenience script (macOS)

This starts everything with one command.

Create a file run.sh

```bash
#!/bin/bash
source .venv/bin/activate
ollama serve &
python ingest.py
streamlit run app.py
```
Then

```bash
chmod +x run.sh
./run.sh
```

