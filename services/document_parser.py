import io
import logging
from datetime import datetime
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("EnterpriseRAG")
ocr_engine = None

def get_ocr_engine():
    """Lazily initializes RapidOCR to prevent app startup crashes."""
    global ocr_engine
    if ocr_engine is None:
        try:
            from rapidocr_onnxruntime import RapidOCR
            ocr_engine = RapidOCR(det_use_cuda=False, rec_use_cuda=False, cls_use_cuda=False)
        except Exception as e:
            logger.warning(f"RapidOCR initialization skipped or failed: {e}")
            ocr_engine = False
    return ocr_engine if ocr_engine is not False else None


class DocumentParser:
    @staticmethod
    def extract_text(file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        pages_text = []
        # Step 1: Digital extraction with pypdf
        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            for i, p in enumerate(reader.pages):
                txt = p.extract_text()
                if txt and len(txt.strip()) > 10:
                    pages_text.append({"text": txt.strip(), "page": i + 1})
        except Exception as e:
            logger.warning(f"Direct text extraction failed on {filename}: {e}")

        # Step 2: OCR Fallback for scanned pages (only if text is empty)
        if not pages_text:
            engine = get_ocr_engine()
            if engine is not None:
                try:
                    import pypdfium2 as pdfium
                    pdf = pdfium.PdfDocument(file_bytes)
                    for i in range(len(pdf)):
                        bitmap = pdf[i].render(scale=1.5).to_pil()
                        buf = io.BytesIO()
                        bitmap.save(buf, format="JPEG", quality=75)
                        res, _ = engine(buf.getvalue())
                        if res:
                            boxes = sorted(res, key=lambda r: (r[0][0][1], r[0][0][0]))
                            pages_text.append({"text": "\n".join([r[1] for r in boxes]), "page": i + 1})
                except Exception as e:
                    logger.error(f"OCR execution failure on {filename}: {e}")
            else:
                logger.info(f"Skipping OCR for {filename} (OCR engine not available).")

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