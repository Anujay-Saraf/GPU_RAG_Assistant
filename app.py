import streamlit as st
import numpy as np
import time
import uuid

from services.vector_store import VectorStoreService, chroma_client, embed_fn
from services.document_parser import DocumentParser
from services.hybrid_retriever import HybridRetriever
from services.reranker import NeuralReranker
from services.llm_service import LLMService, GROUNDED_SYSTEM_PROMPT, GENERAL_FALLBACK_SYSTEM_PROMPT
from config.settings import settings

st.set_page_config(page_title="Enterprise Intelligence Portal", page_icon="⚡", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar Document Ingestion
with st.sidebar:
    st.title("⚙️ Configuration")
    provider = st.selectbox("LLM Provider", ["gemini", "openrouter", "openai"], index=0)
    api_key = st.text_input("API Key", type="password")
    if api_key:
        settings.active_provider = provider
        settings.api_key = api_key

    st.subheader("📄 Knowledge Ingestion")
    uploaded_files = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
    if st.button("Index Documents", use_container_width=True) and uploaded_files:
        with st.spinner("Processing & Indexing PDFs..."):
            all_chunks = []
            for f in uploaded_files:
                pages = DocumentParser.extract_text(f.read(), f.name)
                all_chunks.extend(DocumentParser.chunk_document(pages, f.name))
            
            if all_chunks:
                col = VectorStoreService.get_collection()
                try:
                    chroma_client.delete_collection(name=col.name)
                except Exception:
                    pass
                col = chroma_client.create_collection(name=col.name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})
                
                batch_size = 128
                for i in range(0, len(all_chunks), batch_size):
                    batch = all_chunks[i:i + batch_size]
                    col.upsert(
                        documents=[c["text"] for c in batch],
                        metadatas=[c["meta"] for c in batch],
                        ids=[f"chk_{j}" for j in range(i, i + len(batch))]
                    )
                HybridRetriever.update_bm25_cache()
                st.success(f"Successfully indexed {len(all_chunks)} chunks!")

# Main Chat Interface
st.title("⚡ Enterprise Knowledge Assistant")
st.caption("Grounded Multi-Stage RAG | Live Verification Audits")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("mode") == "general_knowledge":
            st.info("💡 **Answered from General AI Knowledge** *(Not found in local documents)*")
        st.markdown(msg["content"])
        if msg.get("metrics_md"):
            with st.expander("📚 Citation & Confidence Audit"):
                st.markdown(msg["metrics_md"])

prompt = st.chat_input("Ask a question across your indexed documents...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        status_box = st.status("Processing...", expanded=True)
        resp_ph = st.empty()
        
        # 1. Catalog Check
        if any(w in prompt.lower() for w in ["which notes", "what documents", "list files", "which all notes"]):
            summary = VectorStoreService.get_catalog_summary() or "The knowledge base is empty."
            status_box.update(label="✅ Catalog Found", state="complete")
            resp_ph.markdown(summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            st.stop()

        # 2. Retrieval & Re-ranking
        status_box.update(label="🔍 Retrieving context...", state="running")
        sub_queries = HybridRetriever.expand_query(prompt)
        candidates = HybridRetriever.search(sub_queries)
        
        cands, raw_logits, sigmoid_probs = NeuralReranker.rerank(prompt, candidates)
        max_confidence = float(np.max(sigmoid_probs)) if len(sigmoid_probs) > 0 else 0.0
        
        full_response = ""
        final_mode = "grounded_rag"
        metrics_md = ""

        # 3. Generation
        status_box.update(label="✍️ Generating response...", state="running")
        if max_confidence < 0.15:
            final_mode = "general_knowledge"
            st.info("💡 **Answered from General AI Knowledge** *(Topic not found in indexed files)*")
            for token in LLMService.stream_generate(GENERAL_FALLBACK_SYSTEM_PROMPT, f"Question: {prompt}\n\nAnswer:"):
                full_response += token
                resp_ph.markdown(full_response + "▌")
        else:
            ranked_idx = [i for i in sigmoid_probs.argsort()[::-1] if sigmoid_probs[i] >= 0.15][:4]
            final_cands = [cands[i] for i in ranked_idx]
            context_str = "\n\n".join([f"Source [{i+1}] (from {c['meta'].get('source')}, Page {c['meta'].get('page','?')}):\n{c['text']}" for i, c in enumerate(final_cands)])
            
            is_code = LLMService.is_code_request(prompt)
            user_prompt = f"Context Sources:\n{context_str}\n\nQuestion: {prompt}\n\nCited Answer:"
            
            for token in LLMService.stream_generate(GROUNDED_SYSTEM_PROMPT, user_prompt):
                full_response += token
                resp_ph.markdown(full_response + "▌")

            sources_payload = [{
                "index": i + 1,
                "source": c["meta"].get("source"),
                "page": c["meta"].get("page", "N/A"),
                "confidence": f"{round(float(sigmoid_probs[ranked_idx[i]]) * 100, 1)}%",
                "excerpt": c["text"][:180] + "..."
            } for i, c in enumerate(final_cands)]

            overall_conf = round(float(np.mean([float(s["confidence"].replace("%", "")) for s in sources_payload])), 1)
            metrics_md = f"**Overall Confidence:** `{overall_conf}%`\n\n### 📚 Sources:\n"
            for s in sources_payload:
                metrics_md += f"- **[{s['index']}] {s['source']} (p. {s['page']})** — `{s['confidence']}`\n"

        status_box.update(label="✅ Complete!", state="complete")
        resp_ph.markdown(full_response)
        if metrics_md:
            with st.expander("📚 Citation & Confidence Audit", expanded=True):
                st.markdown(metrics_md)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "mode": final_mode,
            "metrics_md": metrics_md
        })