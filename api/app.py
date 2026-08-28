from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes_rag import router as rag_router
from api.routes_admin import router as admin_router
from core.hardware import DEVICE

app = FastAPI(title="Enterprise Decoupled RAG", version="4.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

app.include_router(rag_router)
app.include_router(admin_router)

@app.get("/healthz")
def health():
    return {"status": "healthy", "device": DEVICE}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=False, workers=1)