import io
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from pypdf import PdfReader
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("EnterpriseRAG")
ocr_engine = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)

class DocumentParser:
    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        pages_text = []
        # Step 1: Direct digital extraction
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, p in enumerate(reader.pages):
                txt = p.extract_text()
                if txt and len(txt.strip()) > 10:
                    pages_text.append({"text": txt.strip(), "page": i + 1})
        except Exception:
            pass

        # Step 2: Fallback OCR for scanned pages
        if not pages_text:
            try:
                pdf = pdfium.PdfDocument(file_bytes)
                for i in range(len(pdf)):
                    bitmap = pdf[i].render(scale=1.5).to_pil()
                    buf = io.BytesIO()
                    bitmap.save(buf, format="JPEG", quality=75)
                    res, _ = ocr_engine(buf.getvalue())
                    if res:
                        boxes = sorted(res, key=lambda r: (r[0][0][1], r[0][0][0]))
                        pages_text.append({"text": "\n".join([r[1] for r in boxes]), "page": i + 1})
            except Exception as e:
                logger.error(f"OCR failure on {filename}: {e}")

        return pages_text

    @staticmethod
    def chunk_document(pages_text: List[Dict[str, Any]], filename: str) -> List[Dict[str, Any]]:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=64)
        chunks = []
        for p in pages_text:
            for s in splitter.split_text(p["text"]):
                if s.strip():
                    chunks.append({
                        "text": s,
                        "meta": {
                            "source": filename,
                            "page": p["page"],
                            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        }
                    })
        return chunks