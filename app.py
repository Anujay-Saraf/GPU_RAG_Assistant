import io
import json
import logging
import os
import re
import time
import urllib.request
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
# Global Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Intelligence Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -------------------------------------------------------------
# Secret Resolution Helper
# -------------------------------------------------------------
def get_secret(key: str, default: str = "") -> str:
  try:
    if key in st.secrets:
      return str(st.secrets[key])
  except Exception:
    pass
  return os.getenv(key, default)


ADMIN_PASSKEY = get_secret("ADMIN_SECRET_KEY", "admin-enterprise-key-2026")


# -------------------------------------------------------------
# Dynamic Auto-Free Model Discovery for OpenRouter
# -------------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def get_dynamic_free_models() -> list:
  """Queries OpenRouter API for currently active 100% free model slugs."""
  try:
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "EnterpriseRAG/1.0"},
    )
    with urllib.request.urlopen(req, timeout=4.0) as resp:
      if resp.status == 200:
        data = json.loads(resp.read().decode())
        models = data.get("data", [])
        free_slugs = []
        for m in models:
          model_id = m.get("id", "")
          pricing = m.get("pricing", {})
          is_zero = (
              str(pricing.get("prompt", "")).strip() in ["0", "0.0"]
              and str(pricing.get("completion", "")).strip() in ["0", "0.0"]
          )
          if model_id.endswith(":free") or is_zero:
            if not any(
                bad in model_id.lower()
                for bad in ["rerank", "embed", "guard", "moderation"]
            ):
              free_slugs.append(model_id)
        if free_slugs:
          return free_slugs
  except Exception as e:
    logger.warning(f"OpenRouter models catalog lookup failed: {e}")

  return [
      "google/gemini-2.0-flash-exp:free",
      "meta-llama/llama-3.2-3b-instruct:free",
      "meta-llama/llama-3.1-8b-instruct:free",
      "mistralai/mistral-7b-instruct:free",
  ]


# -------------------------------------------------------------
# Cached Neural Models & Vector Database
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
    show_spinner="Loading Neural Embedding & Re-Ranking Engines..."
)
def load_neural_models():
  embedder = SentenceTransformer("all-MiniLM-L6-v2")
  reranker = CrossEncoder(
      "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512
  )
  chroma_client = chromadb.Client()
  embed_fn = FastEmbedding(embedder)

  collection = chroma_client.get_or_create_collection(
      name="general_knowledge_base",
      embedding_function=embed_fn,
      metadata={"hnsw:space": "cosine"},
  )
  return embedder, reranker, chroma_client, collection, embed_fn


embedder, reranker, chroma_client, collection, embed_fn = load_neural_models()


# -------------------------------------------------------------
# Knowledge Base State & Ingestion Engine
# -------------------------------------------------------------
def get_indexed_inventory():
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
            "uploader": m.get("uploader", "system"),
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
    logger.warning(f"BM25 index update warning: {e}")


def ingest_pdf_files(
    uploaded_files: list, uploader_role: str = "user"
) -> tuple[int, list]:
  splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64)
  indexed_count = 0
  skipped_files = []
  current_inventory = get_indexed_inventory()

  for f in uploaded_files:
    file_bytes = f.read()
    file_size = len(file_bytes)

    # Duplicate check: exact filename and byte size
    if (
        f.name in current_inventory
        and current_inventory[f.name]["file_size"] == file_size
    ):
      skipped_files.append(f.name)
      continue

    # Purge existing older chunks if updated
    if f.name in current_inventory:
      try:
        collection.delete(where={"source": f.name})
      except Exception:
        pass

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
                    "uploader": uploader_role,
                },
            })
    except Exception as e:
      logger.error(f"Error parsing {f.name}: {e}")
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

  update_bm25_index()
  return indexed_count, skipped_files


# -------------------------------------------------------------
# Multi-Tiered Credential & Provider Resolver
# -------------------------------------------------------------
def resolve_effective_provider_and_key() -> tuple[str, str, str]:
  """Resolves active provider and API key with hierarchical priority:

  1. User Session Override (BYOK)
  2. Admin Session Override
  3. Streamlit Cloud Secrets / Environment Variables
  """
  user_override_prov = st.session_state.get("user_selected_provider")
  user_key = st.session_state.get("user_custom_api_key", "").strip()

  if user_override_prov and user_key:
    return user_override_prov, user_key, "User BYOK Session"

  admin_prov = st.session_state.get("admin_active_provider")
  if admin_prov:
    admin_key = st.session_state.get(f"admin_{admin_prov}_key", "").strip()
    if admin_key:
      return admin_prov, admin_key, "Admin Session Override"

  system_prov = admin_prov or get_secret("ACTIVE_PROVIDER", "openrouter")
  system_key = ""
  if system_prov == "openrouter":
    system_key = get_secret("OPENROUTER_API_KEY")
  elif system_prov == "gemini":
    system_key = get_secret("GEMINI_API_KEY")
  elif system_prov == "openai":
    system_key = get_secret("OPENAI_API_KEY")

  if system_key:
    return system_prov, system_key, "System Default"

  return system_prov, "", "Extractive Mode (No Key)"


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
  bm25_data = st.session_state.get(
      "bm25_cache", {"bm25": None, "docs": [], "metas": []}
  )
  if not bm25_data["bm25"]:
    update_bm25_index()
    bm25_data = st.session_state.get(
        "bm25_cache", {"bm25": None, "docs": [], "metas": []}
    )

  where_filter = None
  if selected_sources:
    if len(selected_sources) == 1:
      where_filter = {"source": selected_sources[0]}
    else:
      where_filter = {"source": {"$in": selected_sources}}

  candidates = {}
  for sq in sub_queries:
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
  if not history or len(history) < 2 or not api_key:
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
    if provider == "gemini":
      genai.configure(api_key=api_key)
      model = genai.GenerativeModel("gemini-1.5-flash")
      res = model.generate_content(prompt)
      return res.text.strip().strip('"')
    elif provider in ["openai", "openrouter"]:
      base_url = (
          "https://openrouter.ai/api/v1" if provider == "openrouter" else None
      )
      client = openai.OpenAI(api_key=api_key, base_url=base_url)
      available_models = (
          ["gpt-4o-mini"]
          if provider == "openai"
          else get_dynamic_free_models()
      )
      for model_name in available_models:
        try:
          res = client.chat.completions.create(
              model=model_name,
              messages=[{"role": "user", "content": prompt}],
              max_tokens=60,
          )
          return res.choices[0].message.content.strip().strip('"')
        except Exception:
          continue
  except Exception:
    return latest_query
  return latest_query


def stream_llm(
    provider: str, api_key: str, system_prompt: str, user_prompt: str
):
  if not api_key:
    raise ValueError("No API key configured.")

  if provider == "gemini":
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        "gemini-1.5-flash", system_instruction=system_prompt
    )
    response = model.generate_content(user_prompt, stream=True)
    for chunk in response:
      if chunk.text:
        yield chunk.text

  elif provider == "openai":
    client = openai.OpenAI(api_key=api_key)
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
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

  elif provider == "openrouter":
    client = openai.OpenAI(
        api_key=api_key, base_url="https://openrouter.ai/api/v1"
    )
    live_free_models = get_dynamic_free_models()
    streamed = False

    for candidate_model in live_free_models:
      try:
        stream = client.chat.completions.create(
            model=candidate_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            stream=True,
        )
        for chunk in stream:
          if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
            streamed = True
        if streamed:
          break
      except Exception as model_err:
        logger.warning(
            f"OpenRouter candidate [{candidate_model}] failed: {model_err}"
        )
        continue

    if not streamed:
      raise RuntimeError(
          "All free models are currently unavailable on OpenRouter."
      )


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


# =============================================================
# ROUTE 1: USER PORTAL (/user)
# =============================================================
def render_user_page():
  if "user_messages" not in st.session_state:
    st.session_state.user_messages = []

  # Sidebar: Document Upload + BYOK Settings + Target Selector
  with st.sidebar:
    # 1. User API Key & Provider Override (BYOK)
    with st.expander("🔑 BYOK: Custom API Key (Optional)", expanded=False):
      st.caption("Override system credentials for your session:")
      user_provider_choice = st.selectbox(
          "Your LLM Provider",
          ["openrouter", "gemini", "openai"],
          key="user_prov_select",
          format_func=lambda x: (
              "OpenRouter (Auto Free)"
              if x == "openrouter"
              else ("Google Gemini" if x == "gemini" else "OpenAI")
          ),
      )
      st.session_state.user_selected_provider = user_provider_choice

      user_key_input = st.text_input(
          f"{user_provider_choice.capitalize()} API Key",
          type="password",
          placeholder="sk-...",
          key="user_key_input",
      )
      if user_key_input:
        st.session_state.user_custom_api_key = user_key_input

      if st.button("Clear My Session Key", use_container_width=True):
        st.session_state.pop("user_custom_api_key", None)
        st.session_state.pop("user_selected_provider", None)
        st.rerun()

    # Active Credential Badge
    active_prov, active_key, key_src = resolve_effective_provider_and_key()
    if active_key:
      st.success(f"🟢 Engine: **{active_prov.upper()}** ({key_src})")
    else:
      st.info("🟡 Engine: **Extractive Fallback Mode** (No API Key)")

    st.markdown("---")

    # 2. Document Upload for Real-Time Ingestion
    st.subheader("📤 Upload Documents")
    user_uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        key="user_file_uploader",
    )

    if (
        st.button(
            "⚡ Index My Documents",
            use_container_width=True,
            key="user_index_btn",
        )
        and user_uploaded_files
    ):
      with st.status("Ingesting & Indexing PDF(s)...", expanded=True) as box:
        count, skipped = ingest_pdf_files(
            user_uploaded_files, uploader_role="user"
        )
        box.update(
            label=(
                f"Done! ({count} chunks indexed, {len(skipped)} duplicates"
                " bypassed)"
            ),
            state="complete",
        )
        st.success(f"Added {len(user_uploaded_files)} document(s) to RAG!")
        st.rerun()

    st.markdown("---")

    # 3. Target Document Selector
    st.subheader("📚 Available Knowledge Base")
    current_inventory = get_indexed_inventory()

    if current_inventory:
      doc_names = list(current_inventory.keys())
      selected_docs = st.multiselect(
          "Filter Search Targets:",
          options=doc_names,
          default=doc_names,
          help="Choose which documents the assistant will search against.",
      )
      for name, info in current_inventory.items():
        size_kb = (
            round(info["file_size"] / 1024, 1)
            if info["file_size"] > 0
            else "N/A"
        )
        st.caption(
            f"📄 **{name}** (~{info['pages']} pgs | {info['chunks']} chunks |"
            f" {size_kb} KB)"
        )
    else:
      selected_docs = []
      st.info("No documents currently indexed. Upload above to begin!")

  # -------------------------------------------------------------
  # Parallel Dual-Column Layout: Chat on Left, Citations on Right
  # -------------------------------------------------------------
  col_chat, col_citations = st.columns([3, 2], gap="large")

  with col_chat:
    st.title("⚡ Knowledge Chat")
    st.caption("Grounded Multi-Document RAG Workspace")

    # Render Chat History
    for msg in st.session_state.user_messages:
      with st.chat_message(msg["role"]):
        if msg.get("mode") == "general_knowledge":
          st.info(
              "💡 **Answered from General AI Knowledge** *(Topic not found in"
              " selected documents)*"
          )
        elif msg.get("mode") == "extractive_fallback":
          st.warning(
              "ℹ️ **Extractive Fallback Mode:** Direct verified passages from"
              " your documents."
          )
        st.markdown(msg["content"])

  with col_citations:
    st.title("📚 Citations & Audits")
    st.caption("Live Evidence Verification (Wrapped by Default)")

    # Render previous citation audits inside collapsed expanders
    assistant_turns = [
        m
        for m in st.session_state.user_messages
        if m["role"] == "assistant" and m.get("metrics_md")
    ]
    if not assistant_turns:
      st.info(
          "Citations and evidence audits will appear here in parallel as you"
          " query your documents."
      )
    else:
      for idx, turn in enumerate(assistant_turns, 1):
        q_label = turn.get("query_preview", f"Turn #{idx}")
        # Kept collapsed (expanded=False) until user explicitly clicks to view
        with st.expander(f"🔍 Citation Audit: {q_label}", expanded=False):
          st.markdown(turn["metrics_md"])

  # Input Box
  prompt = st.chat_input("Ask a question across your indexed documents...")

  if prompt:
    # Append user question to history
    st.session_state.user_messages.append({"role": "user", "content": prompt})

    with col_chat:
      with st.chat_message("user"):
        st.markdown(prompt)

      with st.chat_message("assistant"):
        status_box = st.status("Analyzing...", expanded=True)
        notice_ph = st.empty()
        resp_ph = st.empty()

        full_response = ""
        metrics_md = ""
        final_mode = "grounded_rag"

        # 1. Resolve Effective Provider & API Key
        active_prov, active_key, key_src = resolve_effective_provider_and_key()

        # 2. Catalog Lookup Interceptor
        catalog_patterns = [
            r"which (all )?(notes|docs|documents|files|pdfs)",
            r"what (notes|docs|files) (do we have|exist)",
            r"list (all )?(the )?notes",
        ]
        if any(re.search(p, prompt.lower()) for p in catalog_patterns):
          inv = get_indexed_inventory()
          if not inv:
            summary = (
                "The knowledge base is currently empty. Please upload PDF files"
                " in the sidebar."
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
          st.session_state.user_messages.append({
              "role": "assistant",
              "content": summary,
              "mode": "catalog",
              "query_preview": prompt[:40] + "...",
          })
          st.rerun()

        # 3. Contextualization & Sub-Queries
        status_box.update(label="🧠 Expanding query intent...", state="running")
        standalone_query = contextualize_pronouns(
            st.session_state.user_messages[:-1], prompt, active_prov, active_key
        )
        sub_queries = expand_query(standalone_query)

        if collection.count() == 0:
          status_box.update(label="⚠️ Knowledge Base Empty", state="error")
          resp_ph.error(
              "No documents have been indexed yet. Please upload PDF files in"
              " the sidebar."
          )
          st.stop()

        # 4. Hybrid Search & Neural Re-Ranking
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

        # 5. Adaptive Generation with Guaranteed Extractive Fallback
        t_gen = time.time()
        generation_succeeded = False

        if active_key:
          status_box.update(
              label=(
                  f"✍️ Streaming verified response via {active_prov.upper()}..."
              ),
              state="running",
          )
          try:
            if max_confidence < 0.15:
              final_mode = "general_knowledge"
              notice_ph.info(
                  "💡 **Answered from General AI Knowledge** *(Topic not found"
                  " in selected documents)*"
              )
              user_prompt = (
                  f"Topic/Task: {standalone_query}\n\nTechnical Explanation &"
                  " Code:"
              )
              for chunk in stream_llm(
                  active_prov,
                  active_key,
                  GENERAL_FALLBACK_SYSTEM_PROMPT,
                  user_prompt,
              ):
                full_response += chunk
                resp_ph.markdown(full_response + "▌")
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

              user_prompt = (
                  f"Context Sources:\n{context_str}\n\nTask: Provide clean,"
                  " working Python code for"
                  f" '{standalone_query}' based on context.\nInclude inline"
                  " citations [1], [2].\n\nAnswer:"
                  if is_code
                  else (
                      f"Context Sources:\n{context_str}\n\nQuestion:"
                      f" {standalone_query}\n\nCited Answer:"
                  )
              )

              for chunk in stream_llm(
                  active_prov, active_key, GROUNDED_SYSTEM_PROMPT, user_prompt
              ):
                full_response += chunk
                resp_ph.markdown(full_response + "▌")

            gen_time = time.time() - t_gen
            generation_succeeded = True
            status_box.update(label="✅ Complete!", state="complete")
            resp_ph.markdown(full_response)
          except Exception as gen_err:
            logger.warning(
                f"LLM generation failed ({gen_err}). Triggering Extractive"
                " Fallback..."
            )
            full_response = ""

        # PATH: Extractive Fallback (No key or LLM error)
        if not generation_succeeded:
          final_mode = "extractive_fallback"
          status_box.update(
              label="⚡ Extractive Mode (Local Neural Engine)", state="complete"
          )

          if active_key:
            notice_ph.warning(
                "⚠️ **LLM Connection Alert:** Cloud generation encountered an"
                " issue. Displaying direct verified passages from your"
                " documents below."
            )
          else:
            notice_ph.info(
                "ℹ️ **Extractive Mode:** Direct top-ranked passages verified by"
                " local neural cross-encoder."
            )

          if max_confidence >= 0.15:
            ranked_idx = [
                i
                for i in sigmoid_probs.argsort()[::-1]
                if sigmoid_probs[i] >= 0.15
            ][:4]
            lines = [
                f"### 📖 Relevant Excerpts for: *\"{standalone_query}\"*\n"
            ]
            for rank, i in enumerate(ranked_idx, 1):
              c = candidates[i]
              src = c["meta"].get("source", "Document")
              pg = c["meta"].get("page", "?")
              conf = round(float(sigmoid_probs[i]) * 100, 1)
              lines.append(
                  f"**[{rank}] {src} (Page {pg})** — *{conf}% relevance*\n> "
                  + c["text"].strip().replace("\n", "\n> ")
                  + "\n"
              )

            full_response = "\n".join(lines)
            resp_ph.markdown(full_response)
          else:
            full_response = (
                "No closely matching sections found in the selected documents."
            )
            resp_ph.info(full_response)

          gen_time = 0.0

        # 6. Build Grounding Audit String for Right-Side Panel
        if max_confidence >= 0.15:
          ranked_idx = [
              i
              for i in sigmoid_probs.argsort()[::-1]
              if sigmoid_probs[i] >= 0.15
          ][:4]
          final_cands = [candidates[i] for i in ranked_idx]
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
          metrics_md = f"""### 🎯 Grounding Audit
**Answer Confidence:** `{overall_conf}%` ({'🟢 HIGH GROUNDING' if overall_conf >= 70 else '🟡 MODERATE GROUNDING'})

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

        # Record to chat session
        st.session_state.user_messages.append({
            "role": "assistant",
            "content": full_response,
            "mode": final_mode,
            "metrics_md": metrics_md,
            "query_preview": prompt[:40] + ("..." if len(prompt) > 40 else ""),
        })

        # Instant rerun to display parallel citations card in right column
        st.rerun()


# =============================================================
# ROUTE 2: ADMIN CONSOLE (/admin)
# =============================================================
def render_admin_page():
  st.title("👑 Admin Management Console")
  st.caption("Secure Control Center | Document Ingestion & Provider Routing")

  if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False

  # Admin Authentication Gate
  if not st.session_state.admin_logged_in:
    st.subheader("🔐 Admin Access Verification")
    entered_pass = st.text_input(
        "Enter Master Passkey", type="password", key="admin_auth_key"
    )
    if st.button("Unlock Admin Portal", use_container_width=True):
      if entered_pass == ADMIN_PASSKEY:
        st.session_state.admin_logged_in = True
        st.success("Authentication successful.")
        st.rerun()
      else:
        st.error("Invalid passkey.")
    return

  col_a, col_b = st.columns([4, 1])
  with col_b:
    if st.button("🔒 Logout Admin", use_container_width=True):
      st.session_state.admin_logged_in = False
      st.rerun()

  tab1, tab2 = st.tabs(
      ["⚙️ Provider & Credentials", "📤 Document Ingestion & Database"]
  )

  with tab1:
    st.subheader("LLM Provider Configuration")
    providers = ["openrouter", "gemini", "openai"]
    provider_labels = {
        "openrouter": "OpenRouter (Auto Free Router)",
        "gemini": "Google Gemini (gemini-1.5-flash)",
        "openai": "OpenAI (gpt-4o-mini)",
    }
    cur_prov = st.session_state.get(
        "admin_active_provider", get_secret("ACTIVE_PROVIDER", "openrouter")
    )
    selected_prov = st.selectbox(
        "Global Active Provider",
        providers,
        index=providers.index(cur_prov) if cur_prov in providers else 0,
        format_func=lambda x: provider_labels.get(x, x),
    )
    st.session_state.admin_active_provider = selected_prov

    override_key = st.text_input(
        f"{selected_prov.capitalize()} Custom API Key",
        type="password",
        value=st.session_state.get(f"admin_{selected_prov}_key", ""),
        help="Leave empty to use background Streamlit Secrets.",
    )
    if override_key:
      st.session_state[f"admin_{selected_prov}_key"] = override_key
      st.success("Key applied for this active session.")

  with tab2:
    st.subheader("Upload & Index Documents (Admin Authority)")
    admin_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True,
        key="admin_file_uploader",
    )

    if st.button("Process & Index", use_container_width=True) and admin_files:
      with st.status("Indexing documents...", expanded=True) as status_box:
        count, skipped = ingest_pdf_files(admin_files, uploader_role="admin")
        status_box.update(
            label=(
                f"Completed: {count} new chunks indexed,"
                f" {len(skipped)} duplicate files skipped."
            ),
            state="complete",
        )
        st.rerun()

    st.markdown("---")
    st.subheader("Current Database Inventory")
    inv = get_indexed_inventory()
    if inv:
      for name, info in inv.items():
        size_kb = (
            round(info["file_size"] / 1024, 1)
            if info["file_size"] > 0
            else "N/A"
        )
        st.write(
            f"- **`{name}`** (~{info['pages']} pages | {info['chunks']} chunks"
            f" | {size_kb} KB | Tag: `{info.get('uploader', 'system')}`)"
        )

      if st.button("🗑️ Wipe Entire Database", use_container_width=True):
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
        st.success("Vector database cleared.")
        st.rerun()
    else:
      st.info("Database is empty.")


# =============================================================
# URL ROUTER SETUP (st.navigation)
# =============================================================
user_page = st.Page(
    render_user_page,
    title="User Portal",
    icon="💬",
    url_path="user",
    default=True,
)
admin_page = st.Page(
    render_admin_page, title="Admin Console", icon="🔐", url_path="admin"
)

router = st.navigation({
    "Portals": [user_page, admin_page],
})

router.run()