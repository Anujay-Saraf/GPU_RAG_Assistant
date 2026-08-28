import torch
import threading
import logging

logger = logging.getLogger("EnterpriseRAG")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    logger.info(f"🚀 CUDA Acceleration Active: {torch.cuda.get_device_name(0)}")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
else:
    logger.info("⚠️ Running in CPU mode.")

# Global lock to synchronize GPU inferences and prevent VRAM contention
gpu_lock = threading.Lock()