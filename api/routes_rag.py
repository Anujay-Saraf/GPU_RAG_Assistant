import re
import json
import time
import uuid
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import numpy as np

from core.security import SecurityGuardrail
from core.hardware import gpu_lock
from config.settings import settings
from services.vector_store import VectorStoreService, chroma_client, embed_fn
from services.document_parser import DocumentParser
from services.hybrid_retriever import HybridRetriever
from services.reranker import NeuralReranker
from services.llm_service import LLMService, GROUNDED_SYSTEM_PROMPT, GENERAL_FALLBACK_SYSTEM_PROMPT

logger = logging.getLogger("EnterpriseRAG")

router = APIRouter(prefix="/rag", tags=["RAG Pipeline"])
TASK_STORE: Dict[str, Dict[str, Any]] = {}
CHAT_MEMORY: Dict[str, List[Dict[str, str]]] = {}


class QueryReq(BaseModel):
    query: str
    session_id: Optional[str] = "default_session"
    enforce_rerun: Optional[bool] = False
    rerun_instructions: Optional[str] = ""


def bg_ingest(task_id: str, files_data: List[tuple]):
    try:
        TASK_STORE[task_id] = {"status": "processing", "progress": 10, "message": "Parsing document(s)..."}
        all_chunks = []
        for idx, (fname, fbytes) in enumerate(files_data):
            pages = DocumentParser.extract_text(fbytes, fname)
            all_chunks.extend(DocumentParser.chunk_document(pages, fname))
            TASK_STORE[task_id]["progress"] = int(10 + (idx + 1) / len(files_data) * 40)

        if not all_chunks:
            TASK_STORE[task_id] = {"status": "failed", "progress": 100, "message": "No valid text found in PDFs."}
            return

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
        TASK_STORE[task_id] = {"status": "completed", "progress": 100, "message": f"Successfully indexed {len(all_chunks)} chunks."}
    except Exception as e:
        logger.exception("Ingestion failed")
        TASK_STORE[task_id] = {"status": "failed", "progress": 100, "message": str(e)}


@router.post("/ingest/async")
def start_ingest(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    task_id = str(uuid.uuid4())
    data = [(f.filename, f.file.read()) for f in files]
    TASK_STORE[task_id] = {"status": "queued", "progress": 0, "message": "Job queued."}
    background_tasks.add_task(bg_ingest, task_id, data)
    return {"task_id": task_id}


@router.get("/ingest/status/{task_id}")
def check_status(task_id: str):
    if task_id not in TASK_STORE:
        raise HTTPException(status_code=404, detail="Task not found.")
    return TASK_STORE[task_id]


@router.post("/stream")
async def stream_query(req: QueryReq):
    def event_generator():
        session_id = req.session_id
        yield json.dumps({"stage": "intent", "message": "Analyzing query intent and conversational context..."}) + "\n"

        # 1. Catalog / Document Inventory Interceptor
        catalog_patterns = [
            r"which (all )?(notes|docs|documents|files|books|pdfs)",
            r"what (notes|docs|documents|files|books|pdfs) (do we have|are available|exist)",
            r"list (all )?(the )?(notes|docs|documents|files|pdfs)",
            r"share (which all|the) notes",
            r"show (all )?(the )?(notes|documents|files)"
        ]
        if any(re.search(p, req.query.lower().strip()) for p in catalog_patterns):
            summary = VectorStoreService.get_catalog_summary() or "The knowledge base is currently empty. No documents are indexed."
            yield json.dumps({"stage": "intent_done", "intent": "Document Inventory Lookup"}) + "\n"
            for token in summary.split(" "):
                yield json.dumps({"stage": "chunk", "token": token + " "}) + "\n"
                time.sleep(0.01)
            yield json.dumps({
                "stage": "done",
                "mode": "catalog",
                "overall_confidence": "100.0% (🟢 DIRECT CATALOG MATCH)",
                "sources": [],
                "latency": {"retrieval_ms": 1.0, "rerank_ms": 0.0, "generation_sec": 0.1, "total_sec": 0.1}
            }) + "\n"
            return

        # 2. Contextualize Pronouns & Expand Query
        history = CHAT_MEMORY.get(session_id, [])
        standalone_query = req.query if req.enforce_rerun else LLMService.contextualize_query(history, req.query)
        sub_queries = HybridRetriever.expand_query(standalone_query)
        yield json.dumps({"stage": "intent_done", "intent": f"Target: {standalone_query}"}) + "\n"

        col = VectorStoreService.get_collection()
        if col.count() == 0:
            yield json.dumps({"stage": "error", "message": "Knowledge base is empty. Please upload documents."}) + "\n"
            return

        # 3. Multi-Query Hybrid Search
        yield json.dumps({"stage": "retrieving", "message": "Extracting passages across indexed documents..."}) + "\n"
        t0 = time.time()
        candidates = HybridRetriever.search(sub_queries)
        ret_time = time.time() - t0

        if not candidates:
            yield json.dumps({"stage": "error", "message": "No matching candidates found."}) + "\n"
            return

        # 4. Neural Cross-Encoder Re-Ranking & Absolute Sigmoid Calibration
        yield json.dumps({"stage": "reranking", "message": "Evaluating neural cross-attention scores..."}) + "\n"
        t1 = time.time()
        cands, raw_logits, sigmoid_probs = NeuralReranker.rerank(standalone_query, candidates)
        rr_time = time.time() - t1

        max_confidence = float(np.max(sigmoid_probs)) if len(sigmoid_probs) > 0 else 0.0
        t_gen = time.time()
        full_answer = ""
        is_code = LLMService.is_code_request(req.query)

        # 5. Dual-Mode Relevance & Code-Aware Guardrail
        if max_confidence < settings.min_relevance_threshold:
            yield json.dumps({"stage": "generating", "message": "Synthesizing from general domain knowledge..."}) + "\n"
            
            if is_code:
                user_prompt = f"Topic/Task: Provide clean, working Python implementation and code for '{standalone_query}'.\n\nCode & Explanation:"
            else:
                user_prompt = f"Question: {standalone_query}\n\nTechnical Explanation:"

            with gpu_lock:
                for token in LLMService.stream_generate(GENERAL_FALLBACK_SYSTEM_PROMPT, user_prompt):
                    full_answer += token
                    yield json.dumps({"stage": "chunk", "token": token}) + "\n"
            gen_time = time.time() - t_gen

            yield json.dumps({
                "stage": "done",
                "mode": "general_knowledge",
                "overall_confidence": "General AI Knowledge (Not in uploaded files)",
                "sources": [],
                "latency": {
                    "retrieval_ms": round(ret_time * 1000, 1),
                    "rerank_ms": round(rr_time * 1000, 1),
                    "generation_sec": round(gen_time, 2),
                    "total_sec": round(ret_time + rr_time + gen_time, 2)
                }
            }) + "\n"

        else:
            ranked_idx = [i for i in sigmoid_probs.argsort()[::-1] if sigmoid_probs[i] >= settings.min_relevance_threshold][:4]
            if not ranked_idx:
                ranked_idx = sigmoid_probs.argsort()[::-1][:2].tolist()

            final_cands = [cands[i] for i in ranked_idx]
            context_str = "\n\n".join([f"Source [{i+1}] (from {c['meta'].get('source')}, Page {c['meta'].get('page','?')}):\n{c['text']}" for i, c in enumerate(final_cands)])

            prompt = GROUNDED_SYSTEM_PROMPT
            if req.enforce_rerun and req.rerun_instructions:
                prompt += f"\nCRITICAL USER REFINEMENT INSTRUCTION: {req.rerun_instructions}"

            if is_code:
                user_prompt = (
                    f"Context Sources:\n{context_str}\n\n"
                    f"Task: Provide a complete, runnable Python implementation demonstrating the core concepts of '{standalone_query}' based on the principles in the context.\n"
                    f"Write clean Python code with minimal inline citations: [1], [2].\n\nAnswer:"
                )
            else:
                user_prompt = f"Context Sources:\n{context_str}\n\nQuestion: {standalone_query}\n\nCited Answer:"

            yield json.dumps({"stage": "generating", "message": "Streaming verified cited response..."}) + "\n"
            with gpu_lock:
                for token in LLMService.stream_generate(prompt, user_prompt):
                    full_answer += token
                    yield json.dumps({"stage": "chunk", "token": token}) + "\n"
            gen_time = time.time() - t_gen

            sources_payload = [{
                "index": i + 1,
                "source": c["meta"].get("source"),
                "page": c["meta"].get("page", "N/A"),
                "evidence_strength": "🟢 High Relevance" if float(raw_logits[ranked_idx[i]]) >= 0.8 else "🟡 Contextual Match",
                "confidence": f"{round(float(sigmoid_probs[ranked_idx[i]]) * 100, 1)}%",
                "semantic_similarity": f"{round(float(c.get('cosine_sim', 0.50)) * 100, 1)}%",
                "raw_logit": round(float(raw_logits[ranked_idx[i]]), 2),
                "excerpt": c["text"][:180] + "..."
            } for i, c in enumerate(final_cands)]

            overall_conf = round(float(np.mean([float(s["confidence"].replace("%", "")) for s in sources_payload])), 1)
            overall_label = "🟢 HIGH GROUNDING" if overall_conf >= 70 else "🟡 MODERATE GROUNDING"

            yield json.dumps({
                "stage": "done",
                "mode": "grounded_rag",
                "overall_confidence": f"{overall_conf}% ({overall_label})",
                "sources": sources_payload,
                "latency": {
                    "retrieval_ms": round(ret_time * 1000, 1),
                    "rerank_ms": round(rr_time * 1000, 1),
                    "generation_sec": round(gen_time, 2),
                    "total_sec": round(ret_time + rr_time + gen_time, 2)
                }
            }) + "\n"

        # Update Conversation Memory
        CHAT_MEMORY.setdefault(session_id, []).extend([
            {"role": "user", "content": req.query},
            {"role": "assistant", "content": full_answer}
        ])

    return StreamingResponse(event_generator(), media_type="application/x-ndjson")