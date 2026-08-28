import io
import logging
import os
import re
import time
import uuid
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
import google.generativeai as genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import openai
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer
import streamlit as st

logger = logging.getLogger("EnterpriseRAG")

# -------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Intelligence Portal", page_icon="⚡", layout="wide"
)

if "session_id" not in st.session_state:
  st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
  st.session_state.messages = []
if "bm25_cache" not in st.session_state:
  st.session_state.bm25_cache = {"bm25": None, "docs": [], "metas": []}


# -------------------------------------------------------------
# Cached Models & Vector Database
# -------------------------------------------------------------
class FastEmbedding(EmbeddingFunction[Documents]):

  def __init__(self, model):
    self.model = model

  def __call__(self, input: Documents) -> Embeddings:
    return self.model.encode(
        input,
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()


@st.cache_resource(
    show_spinner="Loading Embedding & Re-Ranking Neural Engines..."
)
def load_neural_models():
  embedder = SentenceTransformer("all-MiniLM-L6-v2")
  reranker = CrossEncoder(
      "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
  )
  chroma_client = chromadb.Client()
  embed_fn = FastEmbedding(embedder)

  # Safe initialization: Never crashes on existing collection
  collection = chroma_client.get_or_create_collection(
      name="general_knowledge_base",
      embedding_function=embed_fn,
      metadata={"hnsw:space": "cosine"},
  )
  return embedder, reranker, chroma_client, collection, embed_fn


embedder, reranker, chroma_client, collection, embed_fn = load_neural_models()


# -------------------------------------------------------------
# Knowledge Base State & Helpers
# -------------------------------------------------------------
def get_indexed_inventory():
  """Returns a dictionary of indexed documents with page counts and file sizes."""
  try:
    all_meta = collection.get(include=["metadatas"]).get("metadatas", [])
  except Exception:
    all_meta = []

  doc_summary = {}
  for m in all_meta:
    src = m.get("source")
    if src:
      if src not in doc_summary:
        doc_summary[src] = {
            "pages": int(m.get("page", 1)),
            "file_size": m.get("file_size", 0),
            "chunks": 1,
        }
      else:
        doc_summary[src]["pages"] = max(
            doc_summary[src]["pages"], int(m.get("page", 1))
        )
        doc_summary[src]["chunks"] += 1
        if "file_size" in m:
          doc_summary[src]["file_size"] = m["file_size"]
  return doc_summary


def update_bm25_index():
  try:
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data.get("documents", [])
    if docs:
      tokenized = [re.findall(r"\w+", d.lower()) for d in docs]
      st.session_state.bm25_cache = {
          "bm25": BM25Okapi(tokenized),
          "docs": docs,
          "metas": all_data.get("metadatas", []),
      }
    else:
      st.session_state.bm25_cache = {"bm25": None, "docs": [], "metas": []}
  except Exception as e:
    logger.warning(f"BM25 index update error: {e}")


# -------------------------------------------------------------
# Retrieval & Query Helpers
# -------------------------------------------------------------
ACRONYM_MAP = {
    "dsa": "Data Structures and Algorithms",
    "bst": "Binary Search Tree",
    "oop": "Object Oriented Programming",
    "dbms": "Database Management Systems",
    "os": "Operating Systems",
    "cn": "Computer Networks",
    "ml": "Machine Learning",
    "ds": "Data Science",
    "map": "Maximum A Posteriori",
}


def expand_query(query: str) -> list:
  q_expanded = query
  for acronym, full_text in ACRONYM_MAP.items():
    q_expanded = re.sub(
        r"\b" + re.escape(acronym) + r"\b",
        f"{acronym.upper()} ({full_text})",
        q_expanded,
        flags=re.IGNORECASE,
    )

  sub_queries = [query, q_expanded]
  if any(
      w in query.lower() for w in [" and ", " vs ", " difference ", " compare "]
  ):
    words = re.findall(r"\w+", query.lower())
    main_terms = [
        w
        for w in words
        if w not in ["what", "is", "the", "of", "and", "for", "in", "to", "how"]
    ]
    if len(main_terms) >= 2:
      sub_queries.append(f"{main_terms[0]} concepts")
      sub_queries.append(f"{' '.join(main_terms[1:])} concepts")
  return list(dict.fromkeys(sub_queries))[:3]


def hybrid_search(
    sub_queries: list, selected_sources: list = None, top_k_per_query: int = 8
) -> list:
  bm25_data = st.session_state.bm25_cache
  if not bm25_data["bm25"]:
    update_bm25_index()
    bm25_data = st.session_state.bm25_cache

  where_filter = None
  if selected_sources:
    if len(selected_sources) == 1:
      where_filter = {"source": selected_sources[0]}
    else:
      where_filter = {"source": {"$in": selected_sources}}

  candidates = {}
  for sq in sub_queries:
    # Dense Vector Search
    try:
      res = collection.query(
          query_texts=[sq],
          n_results=min(top_k_per_query, max(1, collection.count())),
          where=where_filter,
          include=["documents", "metadatas", "distances"],
      )
      d_docs = res["documents"][0] if res.get("documents") else []
      d_metas = res["metadatas"][0] if res.get("metadatas") else []
      d_dists = res["distances"][0] if res.get("distances") else []

      for rank, (doc, meta, dist) in enumerate(zip(d_docs, d_metas, d_dists)):
        key = f"{meta.get('source')}_{meta.get('page')}_{doc[:30]}"
        score = 1.0 / (60.0 + rank + 1)
        if key not in candidates:
          candidates[key] = {
              "text": doc,
              "meta": meta,
              "cosine_sim": max(0.0, 1.0 - dist),
              "rrf": score,
          }
        else:
          candidates[key]["rrf"] += score
    except Exception as e:
      logger.warning(f"Vector search warning: {e}")

    # Sparse BM25 Search
    if bm25_data["bm25"]:
      tokens = re.findall(r"\w+", sq.lower())
      scores = bm25_data["bm25"].get_scores(tokens)
      for rank, idx in enumerate(scores.argsort()[::-1][:top_k_per_query]):
        if scores[idx] <= 0:
          continue
        meta = bm25_data["metas"][idx]
        if selected_sources and meta.get("source") not in selected_sources:
          continue

        doc = bm25_data["docs"][idx]
        key = f"{meta.get('source')}_{meta.get('page')}_{doc[:30]}"
        score = 1.0 / (60.0 + rank + 1)
        if key not in candidates:
          candidates[key] = {
              "text": doc,
              "meta": meta,
              "cosine_sim": 0.50,
              "rrf": score,
          }
        else:
          candidates[key]["rrf"] += score

  sorted_cands = sorted(
      candidates.values(), key=lambda x: x["rrf"], reverse=True
  )
  return sorted_cands[:20]


def contextualize_pronouns(
    history: list, latest_query: str, provider: str, api_key: str
) -> str:
  if not history or len(history) < 2:
    return latest_query
  ambiguous = [
      " it",
      " this",
      " that",
      " them",
      " previous",
      " above",
      " implement it",
      " code for it",
  ]
  if not any(t in latest_query.lower() for t in ambiguous):
    return latest_query

  history_str = "\n".join(
      [f"{m['role'].upper()}: {m['content'][:250]}" for m in history[-4:]]
  )
  prompt = f"Rewrite this follow-up question into a standalone search query naming the explicit subject.\nHistory:\n{history_str}\nFollow-up: {latest_query}\nStandalone Query:"

  try:
    if provider == "gemini" and api_key:
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel("gemini-1.5-flash")
      res = model.generate_content(prompt)
      return res.text.strip().strip('"')
    elif provider in ["openai", "openrouter"] and api_key:
      base_url = (
          "https://openrouter.ai/api/v1" if provider == "openrouter" else None
      )
      client = openai.OpenAI(api_key=api_key, base_url=base_url)
      model_name = (
          "gpt-4o-mini"
          if provider == "openai"
          else "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
      )
      res = client.chat.completions.create(
          model=model_name,
          messages=[{"role": "user", "content": prompt}],
          max_tokens=60,
      )
      return res.choices[0].message.content.strip().strip('"')
  except Exception:
    return latest_query
  return latest_query


def stream_llm(
    provider: str, api_key: str, system_prompt: str, user_prompt: str
):
  if not api_key:
    yield (
        "⚠️ **Error:** Missing API Key. Please provide a valid API key in the"
        " sidebar configuration."
    )
    return

  try:
    if provider == "gemini":
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel(
          "gemini-1.5-flash", system_instruction=system_prompt
      )
      response = model.generate_content(user_prompt, stream=True)
      for chunk in response:
        if chunk.text:
          yield chunk.text

    elif provider in ["openai", "openrouter"]:
      base_url = (
          "https://openrouter.ai/api/v1" if provider == "openrouter" else None
      )
      client = openai.OpenAI(api_key=api_key, base_url=base_url)
      model_name = (
          "gpt-4o-mini"
          if provider == "openai"
          else "nvidia/llama-nemotron-rerank-vl-1b-v2:free"
      )
      stream = client.chat.completions.create(
          model=model_name,
          messages=[
              {"role": "system", "content": system_prompt},
              {"role": "user", "content": user_prompt},
          ],
          temperature=0.2,
          stream=True,
      )
      for chunk in stream:
        if chunk.choices[0].delta.content:
          yield chunk.choices[0].delta.content
  except Exception as e:
    yield f"\n\n❌ **API Error ({provider}):** {str(e)}"


# -------------------------------------------------------------
# System Prompts
# -------------------------------------------------------------
GROUNDED_SYSTEM_PROMPT = """You are a grounded academic research assistant. Answer the user question based on the provided Context Sources.

CRITICAL RULES:
1. CITATIONS: Append source numbers [1], [2], etc., directly at the end of factual sentences.
2. Ground your answer strictly in the facts mentioned in the context.
3. If writing code, provide clean, runnable Python code with clear comments.
4. Start directly with the answer/code. Do not write conversational opening phrases."""

GENERAL_FALLBACK_SYSTEM_PROMPT = """You are a senior AI solutions architect.
Answer the user's technical question with precision, working code, and architectural depth.
RULES:
1. Provide ONE clean, working implementation in a single markdown code block.
2. Provide a brief 3-4 line usage example. Avoid repetitive testing boilerplates.
3. Do NOT include bracketed citation numbers like [1] or [2]."""

# -------------------------------------------------------------
# Sidebar: Ingestion & Document Selection
# -------------------------------------------------------------
with st.sidebar:
  st.title("⚙️ Control Center")

  provider = st.selectbox(
      "LLM Provider", ["openrouter", "gemini", "openai"], index=0
  )
  api_key = st.text_input(
      f"{provider.capitalize()} API Key",
      type="password",
      help="Enter your API key.",
  )

  st.markdown("---")

  st.subheader("📚 Available Knowledge Base")
  current_inventory = get_indexed_inventory()

  if current_inventory:
    doc_names = list(current_inventory.keys())
    selected_docs = st.multiselect(
        "Target Documents for Queries:",
        options=doc_names,
        default=doc_names,
        help="Select all or specific files to focus the RAG search.",
    )
    for name, info in current_inventory.items():
      size_kb = (
          round(info["file_size"] / 1024, 1) if info["file_size"] > 0 else "N/A"
      )
      st.caption(
          f"📄 **{name}** (~{info['pages']} pgs | {info['chunks']} chunks |"
          f" {size_kb} KB)"
      )

    if st.button("🗑️ Clear Entire Database", use_container_width=True):
      try:
        chroma_client.delete_collection("general_knowledge_base")
      except Exception:
        pass
      collection = chroma_client.get_or_create_collection(
          name="general_knowledge_base",
          embedding_function=embed_fn,
          metadata={"hnsw:space": "cosine"},
      )
      update_bm25_index()
      st.session_state.bm25_cache = {"bm25": None, "docs": [], "metas": []}
      st.success("Knowledge base cleared.")
      st.rerun()
  else:
    selected_docs = []
    st.info("No documents currently indexed.")

  st.markdown("---")

  st.subheader("📤 Upload New Documents")
  uploaded_files = st.file_uploader(
      "Upload PDF Documents", type=["pdf"], accept_multiple_files=True
  )

  if st.button("Process & Index", use_container_width=True) and uploaded_files:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512, chunk_overlap=64
    )
    indexed_count = 0
    skipped_files = []

    with st.status("Indexing documents...", expanded=True) as status_box:
      for f in uploaded_files:
        file_bytes = f.read()
        file_size = len(file_bytes)

        # Duplicate check: same filename and identical size
        if (
            f.name in current_inventory
            and current_inventory[f.name]["file_size"] == file_size
        ):
          st.write(
              f"⏩ **Bypassed duplicate:** `{f.name}` (Already fully indexed)"
          )
          skipped_files.append(f.name)
          continue

        # If file exists but size changed, delete old chunks first
        if f.name in current_inventory:
          st.write(f"🔄 **Updating modified file:** `{f.name}`...")
          try:
            collection.delete(where={"source": f.name})
          except Exception:
            pass

        st.write(f"📖 Parsing & Chunking `{f.name}`...")
        all_chunks = []
        try:
          reader = PdfReader(io.BytesIO(file_bytes))
          for idx, page in enumerate(reader.pages):
            txt = page.extract_text()
            if txt and len(txt.strip()) > 10:
              for chunk in splitter.split_text(txt.strip()):
                all_chunks.append({
                    "text": chunk,
                    "meta": {
                        "source": f.name,
                        "page": idx + 1,
                        "file_size": file_size,
                    },
                })
        except Exception as e:
          st.error(f"Error parsing {f.name}: {e}")
          continue

        if all_chunks:
          batch_size = 64
          for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            collection.upsert(
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] for c in batch],
                ids=[
                    f"{f.name}_{uuid.uuid4().hex[:8]}_{j}"
                    for j in range(i, i + len(batch))
                ],
            )
          indexed_count += len(all_chunks)
          st.write(f"✅ Indexed `{f.name}` ({len(all_chunks)} chunks)")

      update_bm25_index()
      status_box.update(
          label=(
              f"Done! ({indexed_count} new chunks indexed,"
              f" {len(skipped_files)} skipped)"
          ),
          state="complete",
      )
      st.rerun()

# -------------------------------------------------------------
# Main Chat Workspace
# -------------------------------------------------------------
st.title("⚡ Enterprise Knowledge Assistant")
st.caption(
    "Grounded Hybrid RAG | Multi-Document Selection | In-Memory Confidence"
    " Audits"
)

for msg in st.session_state.messages:
  with st.chat_message(msg["role"]):
    if msg.get("mode") == "general_knowledge":
      st.info(
          "💡 **Answered from General AI Knowledge** *(Not found in selected"
          " indexed files)*"
      )
    st.markdown(msg["content"])
    if msg.get("metrics_md"):
      with st.expander("📚 Evidence & Verification Audit"):
        st.markdown(msg["metrics_md"])

prompt = st.chat_input("Ask a question across your documents...")

if prompt:
  st.session_state.messages.append({"role": "user", "content": prompt})
  with st.chat_message("user"):
    st.markdown(prompt)

  with st.chat_message("assistant"):
    status_box = st.status("Analyzing...", expanded=True)
    notice_ph = st.empty()
    resp_ph = st.empty()

    full_response = ""
    metrics_md = ""
    final_mode = "grounded_rag"

    # 1. Catalog / Document Inventory Check
    catalog_patterns = [
        r"which (all )?(notes|docs|documents|files|pdfs)",
        r"what (notes|docs|files) (do we have|exist)",
        r"list (all )?(the )?notes",
    ]
    if any(re.search(p, prompt.lower()) for p in catalog_patterns):
      inv = get_indexed_inventory()
      if not inv:
        summary = (
            "The knowledge base is currently empty. Please upload PDF files in"
            " the sidebar."
        )
      else:
        lines = ["Here is the inventory of indexed documents:\n"]
        for idx, (doc_name, dinfo) in enumerate(inv.items(), 1):
          lines.append(
              f"- **[{idx}] `{doc_name}`** (~{dinfo['pages']} pages,"
              f" {dinfo['chunks']} chunks indexed)"
          )
        summary = "\n".join(lines)

      status_box.update(label="✅ Inventory Resolved", state="complete")
      resp_ph.markdown(summary)
      st.session_state.messages.append(
          {"role": "assistant", "content": summary, "mode": "catalog"}
      )
      st.stop()

    # 2. Contextualization & Sub-Query Expansion
    status_box.update(label="🧠 Expanding query intent...", state="running")
    standalone_query = contextualize_pronouns(
        st.session_state.messages[:-1], prompt, provider, api_key
    )
    sub_queries = expand_query(standalone_query)

    if collection.count() == 0:
      status_box.update(label="⚠️ Knowledge Base Empty", state="error")
      resp_ph.error(
          "No documents have been indexed yet. Please upload PDF files in the"
          " sidebar."
      )
      st.stop()

    # 3. Filtered Hybrid Search & Neural Re-Ranking
    status_box.update(
        label="🔍 Hybrid search & Cross-Encoder ranking...", state="running"
    )
    t0 = time.time()
    candidates = hybrid_search(sub_queries, selected_sources=selected_docs)
    ret_time = time.time() - t0

    t1 = time.time()
    pairs = [[standalone_query, c["text"]] for c in candidates]
    raw_logits = reranker.predict(pairs) if pairs else np.array([])
    sigmoid_probs = (
        (1.0 / (1.0 + np.exp(-raw_logits)))
        if len(raw_logits) > 0
        else np.array([])
    )
    rr_time = time.time() - t1

    max_confidence = (
        float(np.max(sigmoid_probs)) if len(sigmoid_probs) > 0 else 0.0
    )
    is_code = any(
        t in prompt.lower()
        for t in [
            "code",
            "implement",
            "python",
            "script",
            "function",
            "class",
        ]
    )

    # 4. Dual-Mode Generation
    status_box.update(label="✍️ Streaming verified response...", state="running")
    t_gen = time.time()

    if max_confidence < 0.15:
      final_mode = "general_knowledge"
      notice_ph.info(
          "💡 **Answered from General AI Knowledge** *(Topic not found in"
          " selected documents)*"
      )
      user_prompt = (
          f"Topic/Task: {standalone_query}\n\nTechnical Explanation & Code:"
      )

      for chunk in stream_llm(
          provider, api_key, GENERAL_FALLBACK_SYSTEM_PROMPT, user_prompt
      ):
        full_response += chunk
        resp_ph.markdown(full_response + "▌")

      gen_time = time.time() - t_gen
      metrics_md = f"""**⏱️ Resolution Performance:**
- Hybrid Search: `{round(ret_time * 1000, 1)} ms`
- Cross-Encoder Re-Ranking: `{round(rr_time * 1000, 1)} ms`
- Model Generation: `{round(gen_time, 2)} s`

*Note: Selected documents did not meet relevance threshold. Fallback domain generation used.*"""

    else:
      ranked_idx = [
          i
          for i in sigmoid_probs.argsort()[::-1]
          if sigmoid_probs[i] >= 0.15
      ][:4]
      final_cands = [candidates[i] for i in ranked_idx]
      context_str = "\n\n".join([
          f"Source [{i+1}] (from {c['meta'].get('source')}, Page"
          f" {c['meta'].get('page','?')}):\n{c['text']}"
          for i, c in enumerate(final_cands)
      ])

      if is_code:
        user_prompt = (
            f"Context Sources:\n{context_str}\n\nTask: Provide clean, working"
            f" Python code for '{standalone_query}' based on"
            " context.\nInclude inline citations [1], [2].\n\nAnswer:"
        )
      else:
        user_prompt = (
            f"Context Sources:\n{context_str}\n\nQuestion:"
            f" {standalone_query}\n\nCited Answer:"
        )

      for chunk in stream_llm(
          provider, api_key, GROUNDED_SYSTEM_PROMPT, user_prompt
      ):
        full_response += chunk
        resp_ph.markdown(full_response + "▌")

      gen_time = time.time() - t_gen

      sources_payload = [{
          "index": i + 1,
          "source": c["meta"].get("source"),
          "page": c["meta"].get("page", "N/A"),
          "confidence": f"{round(float(sigmoid_probs[ranked_idx[i]]) * 100, 1)}%",
          "raw_logit": round(float(raw_logits[ranked_idx[i]]), 2),
          "excerpt": c["text"][:180] + "...",
      } for i, c in enumerate(final_cands)]

      overall_conf = round(
          float(
              np.mean([
                  float(s["confidence"].replace("%", ""))
                  for s in sources_payload
              ])
          ),
          1,
      )
      metrics_md = f"""### 🎯 Response Grounding Audit
**Overall Answer Confidence:** `{overall_conf}%` ({'🟢 HIGH GROUNDING' if overall_conf >= 70 else '🟡 MODERATE GROUNDING'})

**⏱️ Latency:** Retrieval: `{round(ret_time * 1000, 1)}ms` | Re-Ranking: `{round(rr_time * 1000, 1)}ms` | Generation: `{round(gen_time, 2)}s`

### 📚 Matched Sources:
"""
      for s in sources_payload:
        metrics_md += f"""
---
#### **Source [{s['index']}] — {s['source']} (Page {s['page']})**
- **Confidence:** `{s['confidence']}` | **Attention Logit:** `{s['raw_logit']}`
- **Cited Excerpt:** *"{s['excerpt']}"*
"""

    status_box.update(label="✅ Complete!", state="complete")
    resp_ph.markdown(full_response)

    if metrics_md:
      with st.expander(
          "📚 Evidence & Verification Audit",
          expanded=(final_mode == "grounded_rag"),
      ):
        st.markdown(metrics_md)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "mode": final_mode,
        "metrics_md": metrics_md,
    })