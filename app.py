import streamlit as st
import os
import re
import time
import hashlib
import tempfile
import traceback
from typing import List, Literal, Optional, Dict, Any, Annotated

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.utilities.tavily_search import TavilySearchAPIWrapper
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_groq import ChatGroq
from langchain import hub
from langgraph.graph import START, END, StateGraph
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field, ConfigDict
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader

# ─────────────────────────────────────────────
# ENVIRONMENT SETUP
# ─────────────────────────────────────────────
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = st.secrets["LANGCHAIN_API_KEY"]
os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# ─────────────────────────────────────────────
# STATE MODEL
# ─────────────────────────────────────────────
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    chat_history: List[BaseMessage] = Field(default_factory=list)
    reformulation_count: int = 0
    current_query: Optional[str] = None
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list)
    generated_answer: Optional[str] = None
    next_step: Optional[str] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)


# ─────────────────────────────────────────────
# SYSTEM INITIALISATION
# ─────────────────────────────────────────────
def initialize_system(
    urls: List[str],
    uploaded_files: List[Any],
    chunk_size: int = 250,
    k: int = 3,
    temperature: float = 0.0,
):
    docs = []

    for url in urls:
        try:
            docs.extend(WebBaseLoader(url).load())
            _log(f"Loaded URL: {url}")
        except Exception as e:
            st.warning(f"⚠️ Could not load {url}: {e}")

    for uf in uploaded_files:
        try:
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.getvalue())
                tmp_path = tmp.name

            if os.path.getsize(tmp_path) == 0:
                st.warning(f"⚠️ {uf.name} is empty — skipped.")
                os.unlink(tmp_path)
                continue

            ext = suffix.lower()
            if ext == ".txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif ext == ".pdf":
                loader = PyPDFLoader(tmp_path)
            elif ext == ".docx":
                loader = Docx2txtLoader(tmp_path)
            else:
                st.warning(f"⚠️ Unsupported file type: {uf.name}")
                os.unlink(tmp_path)
                continue

            file_docs = loader.load()
            for doc in file_docs:
                doc.metadata["source"] = uf.name
            docs.extend(file_docs)
            os.unlink(tmp_path)
            _log(f"Loaded file: {uf.name} ({len(file_docs)} pages)")
        except Exception as e:
            st.warning(f"⚠️ Could not load {uf.name}: {e}")

    if not docs:
        st.info("ℹ️ No custom sources — loading default knowledge base.")
        DEFAULT_URLS = [
            "https://medium.com/@sridevi.gogusetty/rag-vs-graph-rag-llama-3-1-8f2717c554e6",
            "https://medium.com/@sridevi.gogusetty/retrieval-augmented-generation-rag-gemini-pro-pinecone-1a0a1bfc0534",
            "https://medium.com/@sridevi.gogusetty/introduction-to-ollama-run-llm-locally-data-privacy-f7e4e58b37a0",
            "https://ollama.com/library",
            "https://ai.google.dev/docs/gemini_api_overview",
        ]
        for url in DEFAULT_URLS:
            try:
                docs.extend(WebBaseLoader(url).load())
            except Exception as e:
                st.warning(f"⚠️ Default URL failed {url}: {e}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=0
    )
    doc_splits = splitter.split_documents(docs)
    _log(f"Split into {len(doc_splits)} chunks (chunk_size={chunk_size})")

    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embeddings,
    )
    retriever_instance = vectorstore.as_retriever(search_kwargs={"k": k})

    search = TavilySearchAPIWrapper()
    web_search_tool = TavilySearchResults(
        api_wrapper=search,
        max_results=5,
        include_answer=True,
        include_raw_content=True,
        include_images=True,
    )

    # ── Graph ──
    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_agent)
    workflow.add_node("retrieve", retrieve_agent)
    workflow.add_node("reformulate_query", reformulate_agent)
    workflow.add_node("web_search", web_search_agent)
    workflow.add_node("synthesize", synthesize_agent)
    workflow.add_node("generate", generate_agent)
    workflow.add_node("fact_check", fact_check_agent)
    workflow.add_node("safety_check", safety_agent)
    workflow.add_node("ask_clarification", ask_clarification)

    workflow.add_edge(START, "router")
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "retrieve": "retrieve",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification",
            "generate": "generate",
        },
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "relevant": "synthesize",
            "reformulate": "reformulate_query",
            "web_search": "web_search",
            "clarify": "ask_clarification",
        },
    )
    workflow.add_conditional_edges(
        "reformulate_query",
        should_retry_retrieval,
        {"retrieve": "retrieve", "web_search": "web_search"},
    )
    workflow.add_edge("web_search", "synthesize")
    workflow.add_edge("synthesize", "generate")
    workflow.add_edge("generate", "fact_check")
    workflow.add_edge("fact_check", "safety_check")
    workflow.add_edge("safety_check", END)
    workflow.add_edge("ask_clarification", END)

    return workflow.compile(), retriever_instance, web_search_tool, temperature


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def _log(msg: str):
    st.session_state.logs.append(msg)


def _llm(temperature: Optional[float] = None, model: str = "llama-3.3-70b-versatile") -> ChatGroq:
    temp = temperature if temperature is not None else st.session_state.temperature
    return ChatGroq(temperature=temp, model_name=model, groq_api_key=GROQ_API_KEY)


# ─────────────────────────────────────────────
# AGENT NODES
# ─────────────────────────────────────────────
def router_agent(state: AgentState) -> dict:
    _log("── ROUTER AGENT ──")
    model = _llm(temperature=0)
    prompt = PromptTemplate(
        template="""You are the Router Agent. Analyze the user's query and conversation history.

Conversation History (last 5 turns):
{history}

Current Question: {question}

Choose EXACTLY ONE action:
- "retrieve"     → Answer available in internal knowledge base
- "reformulate"  → Query needs rephrasing before internal retrieval
- "web_search"   → Requires real-time / external information
- "clarify"      → Query is ambiguous and needs clarification
- "generate"     → Simple conversational reply; no tools needed

Respond with ONLY the action word, nothing else.""",
        input_variables=["question", "history"],
    )
    history_str = "\n".join(
        [f"{m.type}: {m.content}" for m in state.chat_history[-5:]]
    )
    resp = model.invoke(prompt.format(question=state.current_query, history=history_str))
    decision = resp.content.strip().lower()
    _log(f"Routing → {decision}")
    return {"next_step": decision}


def retrieve_agent(state: AgentState) -> dict:
    _log("── RETRIEVAL AGENT ──")
    try:
        raw_docs = st.session_state.retriever_instance.get_relevant_documents(
            state.current_query
        )
        docs = [{"content": d.page_content, "metadata": d.metadata} for d in raw_docs]
        _log(f"Retrieved {len(docs)} document chunks")
        return {"retrieved_docs": docs}
    except Exception as e:
        _log(f"Retrieval error: {e}")
        return {"retrieved_docs": []}


def reformulate_agent(state: AgentState) -> dict:
    _log("── QUERY REFORMULATION AGENT ──")
    model = _llm(temperature=min(st.session_state.temperature + 0.2, 1.0))
    prompt = PromptTemplate(
        template="""Reformulate the query to improve document retrieval.

Original Query: {query}
Conversation History: {history}
Previous Retrieval Snippet: {previous}

Rules:
- Use synonyms, expand acronyms, split compound questions
- Keep it concise (1–2 sentences max)
- Return ONLY the reformulated query.""",
        input_variables=["query", "history", "previous"],
    )
    history_str = "\n".join(
        [f"{m.type}: {m.content}" for m in state.chat_history[-3:]]
    )
    prev = (
        "\n".join([d["content"] for d in state.retrieved_docs])[:400]
        if state.retrieved_docs
        else "None"
    )
    resp = model.invoke(
        prompt.format(
            query=state.current_query, history=history_str, previous=prev
        )
    )
    new_query = resp.content.strip()
    _log(f"Reformulated: {new_query}")
    return {
        "current_query": new_query,
        "reformulation_count": state.reformulation_count + 1,
    }


def web_search_agent(state: AgentState) -> dict:
    _log("── WEB SEARCH AGENT ──")
    try:
        results = st.session_state.web_search_tool.invoke(
            {"query": state.current_query}
        )
        docs = [
            {
                "content": d["content"],
                "metadata": {"source": d.get("url", "web_search")},
            }
            for d in results
        ]
        _log(f"Web search returned {len(docs)} results")
        return {"retrieved_docs": docs}
    except Exception as e:
        _log(f"Web search error: {e}")
        return {"retrieved_docs": []}


def synthesize_agent(state: AgentState) -> dict:
    _log("── SYNTHESIS AGENT ──")
    if not state.retrieved_docs:
        _log("No documents to synthesize")
        return {"retrieved_docs": []}

    model = _llm()
    prompt = PromptTemplate(
        template="""Synthesize the key information from these sources to answer the question.

Sources:
{sources}

Question: {question}

Instructions:
- Extract only relevant facts
- Cite sources where possible (Source 1, Source 2 …)
- Be concise but comprehensive
- If sources conflict, note the discrepancy""",
        input_variables=["sources", "question"],
    )
    sources_text = "\n\n".join(
        [
            f"[Source {i+1} — {d['metadata'].get('source', 'unknown')}]:\n{d['content']}"
            for i, d in enumerate(state.retrieved_docs)
        ]
    )
    resp = model.invoke(
        prompt.format(sources=sources_text, question=state.current_query)
    )
    _log(f"Synthesized {len(resp.content.split())} words")
    return {
        "retrieved_docs": [
            {"content": resp.content, "metadata": {"source": "synthesized"}}
        ]
    }


def generate_agent(state: AgentState) -> dict:
    _log("── GENERATION AGENT ──")
    if not state.retrieved_docs:
        return {
            "generated_answer": "I don't have enough information to answer that question."
        }
    model = _llm()
    rag_prompt = hub.pull("rlm/rag-prompt")
    chain = rag_prompt | model | StrOutputParser()
    context = "\n\n".join([d["content"] for d in state.retrieved_docs])
    answer = chain.invoke({"context": context, "question": state.current_query})
    _log("Answer generated")
    return {"generated_answer": answer}


def fact_check_agent(state: AgentState) -> dict:
    _log("── FACT-CHECK AGENT ──")
    if not state.generated_answer:
        return {"generated_answer": ""}

    model = _llm(temperature=0)
    claims_resp = model.invoke(
        f"List factual claims as bullet points (start each with '- '):\n\n{state.generated_answer}"
    ).content
    claims = [
        c.strip()
        for c in claims_resp.split("\n")
        if c.strip().startswith("-")
    ][:3]
    _log(f"Verifying {len(claims)} claims")

    verified = []
    for claim in claims:
        try:
            results = st.session_state.web_search_tool.invoke(
                {"query": f"Verify: {claim[2:]}"}
            )
            sources = [d["content"] for d in results][:2]
            verdict = model.invoke(
                f"Is this claim true based on these sources?\n"
                f"Claim: {claim}\nSources: {sources}\n"
                f"Reply with TRUE or FALSE and a brief reason."
            ).content
            verified.append(f"{claim} → {verdict}")
        except Exception as e:
            _log(f"Fact-check failed for: {claim} ({e})")
            verified.append(f"{claim} → Could not verify")

    suffix = (
        "\n\n**🔍 Fact-Check Results:**\n" + "\n".join(verified)
        if verified
        else "\n\n**🔍 Fact-Check:** No verifiable claims identified."
    )
    return {"generated_answer": state.generated_answer + suffix}


def safety_agent(state: AgentState) -> dict:
    _log("── SAFETY AGENT ──")
    if not state.generated_answer:
        return {"generated_answer": ""}

    model = _llm(temperature=0)
    prompt = PromptTemplate(
        template="""Analyse for harmful, biased or inappropriate content.

Text:
{text}

Respond EXACTLY in this format:
Safety Rating: [SAFE|CONCERN|UNSAFE]
Issues: [issues or None]
Revised Text: [revised text, 'Not revisable', or N/A]""",
        input_variables=["text"],
    )
    resp = model.invoke(prompt.format(text=state.generated_answer)).content

    rating_m = re.search(r"Safety Rating:\s*\[?(SAFE|CONCERN|UNSAFE)\]?", resp, re.I)
    rating = rating_m.group(1).upper() if rating_m else "UNKNOWN"

    revised_m = re.search(r"Revised Text:\s*(.*?)(?=\n[A-Z]|$)", resp, re.S | re.I)
    revised = revised_m.group(1).strip() if revised_m else "N/A"

    _log(f"Safety rating: {rating}")

    if rating == "SAFE":
        return {"generated_answer": state.generated_answer}
    if "not revisable" in revised.lower():
        return {
            "generated_answer": "⚠️ I cannot provide a response to that query due to safety concerns."
        }
    if revised and revised != "N/A":
        return {"generated_answer": revised}
    return {"generated_answer": state.generated_answer}


def ask_clarification(state: AgentState) -> dict:
    _log("── CLARIFICATION AGENT ──")
    return {
        "generated_answer": (
            "🤔 I need a bit more context to answer accurately. "
            "Could you rephrase or add more details to your question?"
        )
    }


# ─────────────────────────────────────────────
# DECISION FUNCTIONS
# ─────────────────────────────────────────────
def route_decision(state: AgentState) -> str:
    valid = {"retrieve", "reformulate", "web_search", "clarify", "generate"}
    decision = state.next_step if state.next_step in valid else "retrieve"
    return decision


def grade_documents(
    state: AgentState,
) -> Literal["relevant", "reformulate", "web_search", "clarify"]:
    _log("── GRADING DOCUMENTS ──")
    if not state.retrieved_docs:
        _log("No docs — falling back to web_search")
        return "web_search"

    class Grade(BaseModel):
        score: str = Field(description="relevant | partial | irrelevant")
        action: str = Field(
            description="synthesize | reformulate | web_search | clarify"
        )

    model = _llm(temperature=0, model="llama-3.1-8b-instant").with_structured_output(Grade)
    prompt = PromptTemplate(
        template="""Grade document relevance for the question.

Question: {question}
Documents (truncated): {context}

Score: relevant | partial | irrelevant
Action: synthesize | reformulate | web_search | clarify

Return JSON only: {{"score": "...", "action": "..."}}""",
        input_variables=["question", "context"],
    )
    context = "\n\n".join([d["content"] for d in state.retrieved_docs])[:4000]
    try:
        result = model.invoke(
            prompt.format(question=state.current_query, context=context)
        )
        _log(f"Relevance: {result.score} → Action: {result.action}")
        return "relevant" if result.action == "synthesize" else result.action
    except Exception as e:
        _log(f"Grading error: {e} — defaulting to web_search")
        return "web_search"


def should_retry_retrieval(
    state: AgentState,
) -> Literal["retrieve", "web_search"]:
    if state.reformulation_count < 2:
        _log(f"Reformulation #{state.reformulation_count} — retrying retrieval")
        return "retrieve"
    _log("Reformulation limit reached — switching to web_search")
    return "web_search"


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────
def calculate_knowledge_hash(urls: List[str], files: List[Any]) -> str:
    content = "|".join(sorted(urls))
    for f in files:
        content += f.getvalue().decode(errors="ignore")
    return hashlib.md5(content.encode()).hexdigest()


# ─────────────────────────────────────────────
# STREAMLIT PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Agentic RAG System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --bg: #0d0f14;
    --surface: #161920;
    --surface2: #1e2230;
    --border: #2a2f40;
    --accent: #6c8cff;
    --accent2: #4ade80;
    --accent3: #f59e0b;
    --text: #e2e8f0;
    --muted: #64748b;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: var(--bg); color: var(--text); }

/* Header */
.hero-header {
    background: linear-gradient(135deg, #0d0f14 0%, #1a1f35 50%, #0d1421 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(108,140,255,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(74,222,128,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #6c8cff, #4ade80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.25rem 0;
}
.hero-sub {
    color: var(--muted);
    font-size: 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 0;
}

/* Agent badges */
.badge-row { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1rem; }
.badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    border: 1px solid;
    letter-spacing: 0.04em;
}
.badge-blue { color: #6c8cff; border-color: #6c8cff33; background: #6c8cff11; }
.badge-green { color: #4ade80; border-color: #4ade8033; background: #4ade8011; }
.badge-amber { color: #f59e0b; border-color: #f59e0b33; background: #f59e0b11; }
.badge-red { color: #f87171; border-color: #f8717133; background: #f8717111; }

/* Chat bubbles */
[data-testid="stChatMessage"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    margin-bottom: 0.75rem !important;
}

/* Log code blocks */
.log-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent2);
    padding: 0.2rem 0.5rem;
    border-left: 2px solid var(--accent2);
    margin: 0.2rem 0;
    background: rgba(74,222,128,0.04);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6c8cff22, #4ade8022);
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: var(--accent) !important;
    background: linear-gradient(135deg, #6c8cff33, #4ade8033) !important;
}

/* Progress */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #6c8cff, #4ade80) !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Info/warning boxes */
.stAlert {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    border-radius: 10px !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Hero Header ──
st.markdown(
    """
<div class="hero-header">
  <p class="hero-title">🤖 Advanced Agentic RAG System</p>
  <p class="hero-sub">multi-agent · self-correcting · fact-checked · safe</p>
  <div class="badge-row">
    <span class="badge badge-blue">Router</span>
    <span class="badge badge-blue">Retriever</span>
    <span class="badge badge-blue">Reformulator</span>
    <span class="badge badge-green">Web Search</span>
    <span class="badge badge-green">Synthesizer</span>
    <span class="badge badge-green">Generator</span>
    <span class="badge badge-amber">Fact Checker</span>
    <span class="badge badge-red">Safety Agent</span>
    <span class="badge badge-amber">Clarifier</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# SESSION STATE DEFAULTS
# ─────────────────────────────────────────────
defaults = {
    "logs": [],
    "chat_history": [],
    "final_answer": "",
    "params_applied": False,
    "temperature": 0.0,
    "knowledge_hash": "",
    "chunk_size": 250,
    "retriever_k": 3,
    "total_queries": 0,
    "total_sources": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    with st.expander("Model Parameters", expanded=True):
        chunk_size = st.slider(
            "Chunk Size", 100, 1000,
            st.session_state.chunk_size, 50,
            help="Size of text chunks for embedding"
        )
        retriever_k = st.slider(
            "Top-K Documents", 1, 10,
            st.session_state.retriever_k, 1,
            help="Number of documents retrieved per query"
        )
        temperature = st.slider(
            "LLM Temperature", 0.0, 1.0,
            st.session_state.temperature, 0.1,
            help="Higher = more creative, Lower = more factual"
        )

    st.markdown("---")
    st.markdown("### 📚 Knowledge Sources")
    url_input = st.text_area(
        "URLs (one per line)",
        height=120,
        placeholder="https://example.com/article\nhttps://docs.example.com/guide",
    )
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=["txt", "pdf", "docx"],
        accept_multiple_files=True,
        help="Supports PDF, DOCX, and TXT files",
    )

    apply_btn = st.button("🚀 Apply & Rebuild Index", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 Agent Pipeline")
    st.markdown(
        """
<small>
<b style='color:#6c8cff'>Router</b> — Classifies query intent<br>
<b style='color:#6c8cff'>Retriever</b> — Vector store lookup<br>
<b style='color:#6c8cff'>Reformulator</b> — Query refinement<br>
<b style='color:#4ade80'>Web Search</b> — Real-time Tavily search<br>
<b style='color:#4ade80'>Synthesizer</b> — Multi-source fusion<br>
<b style='color:#4ade80'>Generator</b> — RAG answer generation<br>
<b style='color:#f59e0b'>Fact Checker</b> — Claim verification<br>
<b style='color:#f87171'>Safety Agent</b> — Content safety filter<br>
<b style='color:#f59e0b'>Clarifier</b> — Ambiguity resolution
</small>
""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.logs = []
        st.session_state.final_answer = ""
        st.session_state.total_queries = 0
        st.rerun()

# ─────────────────────────────────────────────
# SYSTEM INIT / REBUILD
# ─────────────────────────────────────────────
urls = [u.strip() for u in url_input.split("\n") if u.strip()]
current_hash = calculate_knowledge_hash(urls, uploaded_files or [])
knowledge_changed = current_hash != st.session_state.knowledge_hash

if apply_btn or not st.session_state.params_applied or knowledge_changed:
    st.session_state.chunk_size = chunk_size
    st.session_state.retriever_k = retriever_k
    st.session_state.temperature = temperature

    with st.spinner("🔧 Building index and compiling agent graph…"):
        try:
            (
                st.session_state.graph,
                st.session_state.retriever_instance,
                st.session_state.web_search_tool,
                st.session_state.temperature,
            ) = initialize_system(
                urls=urls,
                uploaded_files=uploaded_files or [],
                chunk_size=chunk_size,
                k=retriever_k,
                temperature=temperature,
            )
            st.session_state.params_applied = True
            st.session_state.knowledge_hash = current_hash
            st.session_state.total_sources = len(urls) + len(uploaded_files or [])
            st.success("✅ System ready!")
        except Exception as e:
            st.error(f"❌ Initialisation failed: {e}")
            st.stop()

# ─────────────────────────────────────────────
# METRICS ROW
# ─────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric("Queries Processed", st.session_state.total_queries)
m2.metric("Knowledge Sources", st.session_state.total_sources or len(urls) + len(uploaded_files or []))
m3.metric("Chunk Size", st.session_state.chunk_size)
m4.metric("Top-K Docs", st.session_state.retriever_k)

st.markdown("---")

# ─────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────
chat_col, log_col = st.columns([2, 1])

with chat_col:
    st.markdown("#### 💬 Chat")

    # Render history
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.write(msg.content)

    if prompt := st.chat_input("Ask anything about your knowledge sources…"):
        user_msg = HumanMessage(content=prompt)
        st.session_state.chat_history.append(user_msg)
        st.session_state.total_queries += 1

        with st.chat_message("user"):
            st.write(prompt)

        agent_state = AgentState(
            messages=[user_msg],
            chat_history=st.session_state.chat_history,
            current_query=prompt,
            reformulation_count=0,
            retrieved_docs=[],
        )

        with st.chat_message("assistant"):
            with st.spinner("🤖 Running multi-agent workflow…"):
                st.session_state.logs = [f"▶ New query: {prompt}"]
                progress = st.progress(0)
                status = st.empty()

                try:
                    step, max_steps = 0, 15
                    last_updates = {}

                    for output in st.session_state.graph.stream(agent_state):
                        node = list(output.keys())[0]
                        last_updates = output[node]
                        step += 1
                        status.caption(
                            f"⚙️ Executing: **{node.replace('_', ' ').title()}**"
                        )
                        progress.progress(min(step / max_steps, 1.0))
                        time.sleep(0.2)
                        if step >= max_steps:
                            _log("⚠️ Safety break: max steps reached")
                            break

                    progress.empty()
                    status.empty()

                    final_answer = (
                        last_updates.get("generated_answer")
                        or "I could not generate a complete response. Please rephrase."
                    )
                    ai_msg = AIMessage(content=final_answer)
                    st.session_state.chat_history.append(ai_msg)
                    st.session_state.final_answer = final_answer
                    st.write(final_answer)

                except Exception:
                    err = traceback.format_exc()
                    _log(f"ERROR:\n{err}")
                    status.error("❌ Workflow error — check logs")
                    fallback = "Sorry, an error occurred. Please check the Execution Logs tab."
                    st.session_state.chat_history.append(AIMessage(content=fallback))
                    st.write(fallback)

with log_col:
    st.markdown("#### 📋 Live Execution Log")
    log_container = st.container(height=500)
    with log_container:
        if st.session_state.logs:
            for entry in st.session_state.logs:
                st.markdown(
                    f"<div class='log-entry'>{entry}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("Logs will appear here during query execution.")

# ─────────────────────────────────────────────
# WORKFLOW DIAGRAM
# ─────────────────────────────────────────────
st.markdown("---")
with st.expander("🔀 Workflow Architecture", expanded=False):
    diag_col, info_col = st.columns([3, 2])
    with diag_col:
        st.graphviz_chart(
            """
digraph G {
    graph [bgcolor="#0d0f14" fontcolor="#e2e8f0" rankdir=TB]
    node  [shape=box style="rounded,filled" fillcolor="#161920"
           color="#2a2f40" fontcolor="#e2e8f0" fontname="Courier" fontsize=11]
    edge  [color="#2a2f40" fontcolor="#64748b" fontsize=9]

    START [label="▶ START" fillcolor="#1e2230"]
    END   [label="⏹ END"   fillcolor="#1e2230"]

    router           [label="🔀 Router"       color="#6c8cff"]
    retrieve         [label="🗂️  Retrieve"     color="#6c8cff"]
    reformulate_query[label="✏️  Reformulate"  color="#6c8cff"]
    web_search       [label="🌐 Web Search"   color="#4ade80"]
    synthesize       [label="🔗 Synthesize"   color="#4ade80"]
    generate         [label="✨ Generate"     color="#4ade80"]
    fact_check       [label="🔍 Fact Check"   color="#f59e0b"]
    safety_check     [label="🛡️  Safety"       color="#f87171"]
    ask_clarification[label="❓ Clarify"      color="#f59e0b"]

    START -> router
    router -> retrieve          [label="retrieve"]
    router -> reformulate_query [label="reformulate"]
    router -> web_search        [label="web_search"]
    router -> ask_clarification [label="clarify"]
    router -> generate          [label="generate"]

    retrieve -> synthesize       [label="relevant"]
    retrieve -> reformulate_query[label="partial"]
    retrieve -> web_search       [label="irrelevant"]
    retrieve -> ask_clarification[label="ambiguous"]

    reformulate_query -> retrieve  [label="retry (≤2×)"]
    reformulate_query -> web_search[label="fallback"]

    web_search -> synthesize
    synthesize -> generate
    generate   -> fact_check
    fact_check -> safety_check
    safety_check     -> END
    ask_clarification -> END
}
""",
            use_container_width=True,
        )
    with info_col:
        st.markdown("**Active Configuration**")
        st.json(
            {
                "chunk_size": st.session_state.chunk_size,
                "retriever_k": st.session_state.retriever_k,
                "temperature": st.session_state.temperature,
                "main_llm": "llama-3.3-70b-versatile (Groq)",
                "grader_llm": "llama-3.1-8b-instant (Groq)",
                "embeddings": "models/embedding-001 (Google)",
                "vector_store": "ChromaDB",
                "web_search": "Tavily",
                "max_reformulations": 2,
                "max_graph_steps": 15,
            }
        )

        st.markdown("**Active Sources**")
        if urls:
            for u in urls:
                st.markdown(f"- 🔗 [{u[:50]}…]({u})" if len(u) > 50 else f"- 🔗 [{u}]({u})")
        if uploaded_files:
            for f in uploaded_files:
                st.markdown(f"- 📄 {f.name}")
        if not urls and not uploaded_files:
            st.caption("Using default knowledge base")

st.markdown(
    "<br><center><small style='color:#2a2f40'>Advanced Agentic RAG · LangGraph · LangChain · Groq · Tavily · ChromaDB</small></center>",
    unsafe_allow_html=True,
)