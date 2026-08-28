import os
import torch
import chromadb
from typing import List, Dict, Any, Optional
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
from core.hardware import DEVICE

class FastGPUEmbedding(EmbeddingFunction[Documents]):
    def __init__(self, model_name="all-MiniLM-L6-v2", device=DEVICE):
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)
        if device == "cuda":
            self.model.half()

    def name(self) -> str:
        return self.model_name

    def __call__(self, input: Documents) -> Embeddings:
        with torch.inference_mode():
            return self.model.encode(input, batch_size=128, normalize_embeddings=True, show_progress_bar=False).tolist()

embed_fn = FastGPUEmbedding()

def get_chroma_client():
    chroma_host = os.getenv("CHROMA_HOST", None)
    if chroma_host:
        return chromadb.HttpClient(host=chroma_host, port=int(os.getenv("CHROMA_PORT", 8000)))
    return chromadb.PersistentClient(path="./chroma_db_general")

chroma_client = get_chroma_client()

class VectorStoreService:
    @staticmethod
    def get_collection(name: str = "general_knowledge_base"):
        all_cols = chroma_client.list_collections()
        for c in all_cols:
            if c.name == name:
                col = chroma_client.get_collection(name=name, embedding_function=embed_fn)
                if col.count() > 0:
                    return col
        return chroma_client.get_or_create_collection(name=name, embedding_function=embed_fn, metadata={"hnsw:space": "cosine"})

    @staticmethod
    def get_catalog_summary() -> Optional[str]:
        col = VectorStoreService.get_collection()
        all_meta = col.get(include=["metadatas"]).get("metadatas", [])
        if not all_meta:
            return None
        doc_summary = {}
        for m in all_meta:
            src = m.get("source", "Unknown Document")
            doc_summary[src] = max(doc_summary.get(src, 0), int(m.get("page", 1)))
        
        lines = ["Here is the inventory of indexed documents in your knowledge base:\n"]
        for idx, (doc_name, total_pages) in enumerate(doc_summary.items(), 1):
            lines.append(f"- **[{idx}] `{doc_name}`** (~{total_pages} pages indexed)")
        lines.append("\nYou can ask technical questions regarding any topic covered in these documents.")
        return "\n".join(lines)