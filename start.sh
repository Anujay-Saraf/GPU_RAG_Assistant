#!/bin/bash
# 1. Start FastAPI Backend Gateway on local port 8000
uvicorn api.app:app --host 127.0.0.1 --port 8000 &

# 2. Wait for the backend to initialize
sleep 3

# 3. Start Streamlit Frontend on Port 7860 (Hugging Face Default)
streamlit run frontend/app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false