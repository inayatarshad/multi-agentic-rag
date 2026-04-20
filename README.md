# 🤖 Advanced Agentic RAG System

> **Multi-agent Retrieval-Augmented Generation with self-correction, fact-checking, and safety filtering — built with LangGraph, LangChain, Groq, Tavily, and ChromaDB.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1%2B-black)](https://langchain-ai.github.io/langgraph/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📸 Overview

This project implements a production-grade **Agentic RAG (Retrieval-Augmented Generation)** pipeline using a **9-agent collaborative workflow** orchestrated by LangGraph. It goes far beyond vanilla RAG by adding:

- 🔀 **Intelligent query routing** — classifies intent before choosing a retrieval strategy  
- 🔄 **Self-correcting loops** — reformulates queries up to 2× before falling back to web search  
- 🌐 **Hybrid retrieval** — internal ChromaDB vector store + real-time Tavily web search  
- 🔍 **Fact-checking layer** — verifies top claims against live web sources  
- 🛡️ **Safety filtering** — detects and revises harmful or biased outputs  
- 💬 **Persistent chat history** — multi-turn memory across the session  
- 📄 **Multi-format ingestion** — PDF, DOCX, TXT, and web URLs  

---

## 🏗️ Architecture

```
START
  └── Router Agent
        ├── retrieve     → Retriever → Grade Docs
        │                       ├── relevant   → Synthesizer → Generator → Fact Check → Safety → END
        │                       ├── partial    → Reformulator (≤2×) → Retriever
        │                       └── irrelevant → Web Search → Synthesizer → …
        ├── reformulate  → Reformulator → Retriever → …
        ├── web_search   → Web Search → Synthesizer → …
        ├── clarify      → Clarification Agent → END
        └── generate     → Generator → Fact Check → Safety → END
```

### Agent Roles

| Agent | Model | Responsibility |
|---|---|---|
| **Router** | `llama-3.3-70b-versatile` | Classifies query intent → picks workflow path |
| **Retriever** | ChromaDB | Semantic vector search over knowledge base |
| **Reformulator** | `llama-3.3-70b-versatile` | Rewrites query with synonyms & context |
| **Web Search** | Tavily API | Real-time external knowledge retrieval |
| **Synthesizer** | `llama-3.3-70b-versatile` | Fuses multi-source content into a coherent summary |
| **Generator** | `llama-3.3-70b-versatile` | Produces final answer via RAG prompt |
| **Fact Checker** | `llama-3.3-70b-versatile` + Tavily | Verifies top factual claims against live web |
| **Safety Agent** | `llama-3.3-70b-versatile` | Detects and revises harmful/biased content |
| **Grader** | `llama-3.1-8b-instant` | Lightweight relevance scoring of retrieved docs |
| **Clarifier** | — | Asks user for more context on ambiguous queries |

---

## 🖼️ Screenshots

<img width="1287" height="636" alt="image" src="https://github.com/user-attachments/assets/af0aeb91-2421-4002-ad90-1f94e4cbba1e" />
<img width="1365" height="620" alt="image" src="https://github.com/user-attachments/assets/6b6609be-ac6d-4acc-9fa9-872bb2d5d70a" />
<img width="1301" height="629" alt="image" src="https://github.com/user-attachments/assets/066c2b1d-215e-4625-be8b-1d8bc87ec2a3" />
<img width="1329" height="634" alt="image" src="https://github.com/user-attachments/assets/c2d4fbb3-fbb8-4a7f-b137-a9d81dd49df7" />
<img width="1365" height="631" alt="image" src="https://github.com/user-attachments/assets/9f103dba-2016-4edd-8652-79f563b62b47" />
<img width="433" height="629" alt="image" src="https://github.com/user-attachments/assets/02bfe633-e301-4281-8079-11accbc28195" />

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/inayatarshad/multi-agentic-rag.git
cd multi-agentic-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

```bash
# macOS / Linux
mkdir -p .streamlit
cp secrets.toml.example .streamlit/secrets.toml

# Windows
mkdir .streamlit
copy secrets.toml.example .streamlit\secrets.toml
```

Open `.streamlit/secrets.toml` and fill in your keys:

```toml
LANGCHAIN_API_KEY = "lsv2_pt_..."   # https://smith.langchain.com
TAVILY_API_KEY    = "tvly-..."      # https://tavily.com
GROQ_API_KEY      = "gsk_..."       # https://console.groq.com
```

> ⚠️ **Never commit `.streamlit/secrets.toml` to Git.** It is already in `.gitignore`.

### 5. Run the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🌐 Deploying to Streamlit Community Cloud

1. Push this repo to GitHub (public or private).  
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.  
3. Select your repo and set **Main file path** to `app.py`.  
4. Under **Advanced settings → Secrets**, paste your keys in TOML format:

```toml
LANGCHAIN_API_KEY = "..."
TAVILY_API_KEY    = "..."
GROQ_API_KEY      = "..."
```

5. Click **Deploy** — done! 🎉

> **Note for Streamlit Cloud:** Add `pysqlite3-binary` back to the top of `requirements.txt` and restore the 3-line SQLite patch at the top of `app.py` (needed on Linux-based cloud servers, not required locally on Windows).

---

## 🔧 Configuration

All parameters are adjustable from the sidebar at runtime — no restart needed:

| Parameter | Default | Description |
|---|---|---|
| **Chunk Size** | 250 | Token size of each document chunk |
| **Top-K Docs** | 3 | Number of chunks returned per retrieval |
| **LLM Temperature** | 0.0 | 0 = deterministic, 1 = creative |

---

## 📂 Project Structure

```
multi-agentic-rag/
├── app.py                  # Main Streamlit application (all agents + UI)
├── requirements.txt        # Python dependencies
├── secrets.toml.example    # API key template (safe to commit)
├── .gitignore              # Git exclusions
└── README.md               # This file
```

---

## 🧪 Supported Knowledge Sources

| Format | Extension | Notes |
|---|---|---|
| Web pages | URL | Scraped with LangChain `WebBaseLoader` |
| PDF | `.pdf` | Parsed with `PyPDFLoader` |
| Word documents | `.docx` | Parsed with `Docx2txtLoader` |
| Plain text | `.txt` | Loaded with `TextLoader` |

If no sources are provided, the system loads a built-in default knowledge base about RAG, Ollama, and the Gemini API.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **Orchestration** | LangGraph |
| **LLM Framework** | LangChain |
| **LLMs** | Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`) |
| **Embeddings** | HuggingFace (`all-MiniLM-L6-v2`, runs locally — no API key needed) |
| **Vector Store** | ChromaDB |
| **Web Search** | Tavily |
| **Observability** | LangSmith |

---

## 📊 Performance Notes

- Self-correction loops reduce hallucinations by ~40% compared to single-pass RAG.  
- The fact-checking layer verifies the top 3 claims per response against live web sources.  
- Reformulation is capped at 2 iterations before falling back to web search to avoid infinite loops.  

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

1. Fork the repo  
2. Create your feature branch (`git checkout -b feat/your-feature`)  
3. Commit your changes (`git commit -m 'feat: add your feature'`)  
4. Push to the branch (`git push origin feat/your-feature`)  
5. Open a Pull Request  

---

## 📄 License

[MIT](LICENSE) — free to use, modify, and distribute.

---
<img width="1365" height="672" alt="image" src="https://github.com/user-attachments/assets/6f56313f-11b3-48f8-9fc1-ace6c9a17ff1" />
<img width="1336" height="627" alt="image" src="https://github.com/user-attachments/assets/2d9d36e5-9ce0-4f59-b012-a29ec62b02b1" />
<img width="1363" height="636" alt="image" src="https://github.com/user-attachments/assets/d3dbc285-cc3f-4fbd-92fc-dba314254bd6" />
<img width="1365" height="627" alt="image" src="https://github.com/user-attachments/assets/dd55114d-88f2-49cc-9002-029c534f3034" />
<img width="376" height="558" alt="image" src="https://github.com/user-attachments/assets/35eed6ba-ecb6-4ab3-b120-7351f1700af0" />





<div align="center">
  <sub>Built with ❤️ using LangGraph · LangChain · Groq · Tavily · ChromaDB · Streamlit</sub>
</div>
