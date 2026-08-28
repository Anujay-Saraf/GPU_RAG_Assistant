import torch
import numpy as np
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from core.hardware import DEVICE, gpu_lock

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE, max_length=512)

class NeuralReranker:
    @staticmethod
    def rerank(query: str, candidates: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        if not candidates:
            return [], np.array([]), np.array([])
        
        pairs = [[query, c["text"]] for c in candidates]
        with gpu_lock:
            raw_logits = reranker.predict(pairs)
        
        logits_tensor = torch.tensor(raw_logits, dtype=torch.float32)
        sigmoid_probs = (1.0 / (1.0 + torch.exp(-logits_tensor))).numpy()
        return candidates, raw_logits, sigmoid_probs