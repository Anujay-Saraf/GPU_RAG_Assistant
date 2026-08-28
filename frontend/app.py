import streamlit as st
import requests
import json
import uuid
import time
import os
import streamlit as st
import requests
import os

API_BASE = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

# Connection Health Check
@st.cache_data(ttl=5)
def verify_backend_health():
    try:
        r = requests.get(f"{API_BASE}/healthz", timeout=2)
        if r.status_code == 200:
            return True, r.json()
    except Exception as e:
        return False, str(e)
    return False, "Unknown Error"

healthy, info = verify_backend_health()
if not healthy:
    st.error(f"⚠️ **Backend Gateway Offline**: Unable to connect to `{API_BASE}`. Error: {info}")
    st.stop()

st.set_page_config(page_title="Enterprise Intelligence Portal", page_icon="⚡", layout="wide")
API_BASE = os.getenv("BACKEND_API_URL", "http://127.0.0.1:8000")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

nav = st.sidebar.radio("Navigation", ["🔍 Workspace", "⚙️ Control Panel"])

if nav == "🔍 Workspace":
    st.title("⚡ Enterprise Knowledge Assistant")
    st.caption("Decoupled Microservice Architecture | Calibrated Grounding Audits")

    with st.sidebar:
        st.subheader("📄 Knowledge Ingestion")
        uploaded = st.file_uploader("Upload PDF Documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Index Documents", use_container_width=True) and uploaded:
            files_payload = [("files", (f.name, f.read(), "application/pdf")) for f in uploaded]
            try:
                res = requests.post(f"{API_BASE}/rag/ingest/async", files=files_payload)
                if res.status_code == 200:
                    task_id = res.json()["task_id"]
                    prog = st.progress(0)
                    lbl = st.empty()
                    while True:
                        time.sleep(1.0)
                        status = requests.get(f"{API_BASE}/rag/ingest/status/{task_id}").json()
                        prog.progress(status["progress"] / 100.0)
                        lbl.info(status["message"])
                        if status["status"] in ["completed", "failed"]:
                            if status["status"] == "completed":
                                st.success(status["message"])
                            else:
                                st.error(status["message"])
                            break
            except Exception as e:
                st.error(f"Connection failed: {e}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("mode") == "general_knowledge":
                st.info("💡 **Answered from General AI Knowledge** *(Topic not found in indexed files)*")
            st.markdown(msg["content"])
            if msg.get("metrics_md"):
                with st.expander("📚 Citation & Confidence Audit"):
                    st.markdown(msg["metrics_md"])

    if "rerun_prompt" in st.session_state and st.session_state.rerun_prompt:
        prompt = st.session_state.rerun_prompt
        instructions = st.session_state.rerun_instructions
        st.session_state.rerun_prompt = None
    else:
        prompt = st.chat_input("Ask a question across your indexed documents...")
        instructions = ""

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_box = st.status("Processing...", expanded=True)
            notice_ph = st.empty()
            resp_ph = st.empty()
            full_response, metrics_md, final_mode = "", "", "grounded_rag"

            payload = {"query": prompt, "session_id": st.session_state.session_id, "enforce_rerun": bool(instructions), "rerun_instructions": instructions}
            try:
                with requests.post(f"{API_BASE}/rag/stream", json=payload, stream=True) as r:
                    for line in r.iter_lines():
                        if line:
                            data = json.loads(line.decode('utf-8'))
                            stage = data.get("stage")
                            if stage == "intent":
                                status_box.update(label="🧠 Analyzing intent...", state="running")
                            elif stage == "retrieving":
                                status_box.update(label="🔍 Extracting context...", state="running")
                            elif stage == "reranking":
                                status_box.update(label="⚖️ Neural re-ranking...", state="running")
                            elif stage == "generating":
                                status_box.update(label="✍️ Generating response...", state="running")
                            elif stage == "chunk":
                                full_response += data["token"]
                                resp_ph.markdown(full_response + "▌")
                            elif stage == "done":
                                final_mode = data.get("mode", "grounded_rag")
                                status_box.update(label="✅ Response Complete!", state="complete")
                                resp_ph.markdown(full_response)
                                
                                if final_mode == "general_knowledge":
                                    notice_ph.info("💡 **Answered from General AI Knowledge** *(Topic not found in indexed files)*")
                                elif data.get("sources"):
                                    metrics_md = f"**Overall Confidence:** `{data.get('overall_confidence')}`\n\n### 📚 Matched Sources:\n"
                                    for s in data["sources"]:
                                        metrics_md += f"---\n**[{s['index']}] {s['source']} (Page {s['page']})**\n- Confidence: `{s['confidence']}` | Match Score: `{s['semantic_similarity']}`\n- *\"{s['excerpt']}\"*\n"
                                
                                if metrics_md:
                                    with st.expander("📚 Citation & Confidence Audit", expanded=True):
                                        st.markdown(metrics_md)
                                
                                st.session_state.messages.append({"role": "assistant", "content": full_response, "mode": final_mode, "metrics_md": metrics_md})
            except Exception as e:
                st.error(f"Stream error: {e}")

            st.markdown("---")
            c1, c2, _ = st.columns([1, 1, 4])
            with c1:
                if st.button("👍 Satisfied"):
                    st.success("Feedback saved.")
            with c2:
                if st.button("👎 Refine"):
                    st.session_state.show_refine = True

            if st.session_state.get("show_refine"):
                ref_txt = st.text_input("Refinement Instructions:")
                if st.button("Re-run"):
                    st.session_state.show_refine = False
                    st.session_state.rerun_prompt = prompt
                    st.session_state.rerun_instructions = ref_txt
                    st.rerun()

elif nav == "⚙️ Control Panel":
    st.title("⚙️ Enterprise Admin Gateway")
    key = st.text_input("Admin Key", type="password", value="admin-enterprise-key-2026")
    headers = {"X-API-KEY": key}
    if st.button("Authenticate"):
        res = requests.get(f"{API_BASE}/admin/config", headers=headers)
        if res.status_code == 200:
            st.session_state.cfg = res.json()
            st.success("Authenticated.")
        else:
            st.error("Access Denied.")