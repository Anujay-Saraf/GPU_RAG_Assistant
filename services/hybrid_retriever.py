import re
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi
from services.vector_store import VectorStoreService

BM25_CACHE: Dict[str, Any] = {"bm25": None, "docs": [], "metas": []}

ACRONYM_MAP = {
    "dsa": "Data Structures and Algorithms",
    "bst": "Binary Search Tree",
    "oop": "Object Oriented Programming",
    "dbms": "Database Management Systems",
    "os": "Operating Systems",
    "cn": "Computer Networks",
    "ml": "Machine Learning",
    "ds": "Data Science",
    "aml": "Advanced Machine Learning"
}

class HybridRetriever:
    @staticmethod
    def update_bm25_cache():
        col = VectorStoreService.get_collection()
        all_data = col.get(include=["documents", "metadatas"])
        corpus_docs = all_data.get("documents", [])
        if corpus_docs:
            tokenized = [re.findall(r'\w+', doc.lower()) for doc in corpus_docs]
            BM25_CACHE["bm25"] = BM25Okapi(tokenized)
            BM25_CACHE["docs"] = corpus_docs
            BM25_CACHE["metas"] = all_data.get("metadatas", [])
        else:
            BM25_CACHE["bm25"] = None

    @staticmethod
    def expand_query(query: str) -> List[str]:
        q_expanded = query
        for acronym, full_text in ACRONYM_MAP.items():
            q_expanded = re.sub(r'\b' + re.escape(acronym) + r'\b', f"{acronym.upper()} ({full_text})", q_expanded, flags=re.IGNORECASE)

        sub_queries = [query, q_expanded]
        if any(w in query.lower() for w in [" and ", " intersection ", " with ", " vs ", " in "]):
            words = re.findall(r'\w+', query.lower())
            main_terms = [w for w in words if w not in ["what", "is", "the", "of", "and", "for", "in", "to", "how"]]
            if len(main_terms) >= 2:
                sub_queries.append(f"{main_terms[0]} concepts")
                sub_queries.append(f"{' '.join(main_terms[1:])} applications")
        return list(dict.fromkeys(sub_queries))[:3]

    @classmethod
    def search(cls, sub_queries: List[str], top_k_per_query: int = 8) -> List[Dict[str, Any]]:
        col = VectorStoreService.get_collection()
        if not BM25_CACHE["bm25"]:
            cls.update_bm25_cache()

        aggregated_candidates = {}
        for sq in sub_queries:
            # Dense Vector Search
            dense_res = col.query(query_texts=[sq], n_results=min(top_k_per_query, max(1, col.count())), include=["documents", "metadatas", "distances"])
            d_docs = dense_res["documents"][0] if dense_res.get("documents") else []
            d_metas = dense_res["metadatas"][0] if dense_res.get("metadatas") else []
            d_dists = dense_res["distances"][0] if dense_res.get("distances") else []

            for rank, (doc, meta, dist) in enumerate(zip(d_docs, d_metas, d_dists)):
                key = f"{meta.get('source')}_{meta.get('page')}_{doc[:30]}"
                score = 1.0 / (60.0 + rank + 1)
                if key not in aggregated_candidates:
                    aggregated_candidates[key] = {"text": doc, "meta": meta, "cosine_sim": max(0.0, 1.0 - dist), "rrf": score}
                else:
                    aggregated_candidates[key]["rrf"] += score

            # Sparse BM25 Search
            if BM25_CACHE["bm25"]:
                tokens = re.findall(r'\w+', sq.lower())
                sparse_scores = BM25_CACHE["bm25"].get_scores(tokens)
                for rank, idx in enumerate(sparse_scores.argsort()[::-1][:top_k_per_query]):
                    if sparse_scores[idx] <= 0:
                        continue
                    doc = BM25_CACHE["docs"][idx]
                    meta = BM25_CACHE["metas"][idx]
                    key = f"{meta.get('source')}_{meta.get('page')}_{doc[:30]}"
                    score = 1.0 / (60.0 + rank + 1)
                    if key not in aggregated_candidates:
                        aggregated_candidates[key] = {"text": doc, "meta": meta, "cosine_sim": 0.50, "rrf": score}
                    else:
                        aggregated_candidates[key]["rrf"] += score

        sorted_candidates = sorted(aggregated_candidates.values(), key=lambda x: x["rrf"], reverse=True)

        # Cross-Document Diversity
        doc_groups = {}
        for c in sorted_candidates:
            doc_groups.setdefault(c["meta"].get("source", "default"), []).append(c)

        balanced = []
        max_len = max([len(v) for v in doc_groups.values()]) if doc_groups else 0
        for i in range(max_len):
            for src in doc_groups:
                if i < len(doc_groups[src]):
                    balanced.append(doc_groups[src][i])
        return balanced[:20]