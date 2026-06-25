"""
Local Retrieval-Augmented Generation over the agricultural knowledge base.

Build the index:
    python -m src.rag --build

Query:
    python -m src.rag --query "How do I control fall armyworm in maize?"
    python -m src.rag --query "Yadda za a ajiye masara" --hausa
"""
import argparse
import glob
import hashlib
import os
import pickle
import re

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
CHUNK_SIZE = 650
CHUNK_OVERLAP = 120

STOPWORDS = {
    "about", "after", "again", "also", "anything", "before", "could", "does",
    "explain", "from", "give", "have", "help", "into", "more", "need", "please",
    "safe", "safely", "should", "tell", "that", "their", "there", "this", "what",
    "when", "where", "which", "with", "would", "your", "bayani", "karin", "kuma",
    "yana", "ina", "yadda", "wajen", "don", "da", "na", "a", "zan",
}

TOPIC_TERMS = {
    "maize": {"maize", "corn", "masara"},
    "rice": {"rice", "shinkafa"},
    "cassava": {"cassava", "rogo"},
    "storage": {"store", "storage", "stored", "harvest", "post", "aflatoxin", "dry", "drying", "ajiye", "girbi", "bushe"},
    "fall_armyworm": {"fall", "armyworm", "tsutsa"},
    "fertilizer": {"fertilizer", "fertiliser", "taki", "npk", "urea"},
    "cowpea": {"cowpea", "wake"},
    "groundnut": {"groundnut", "peanut"},
    "tomato": {"tomato", "tumatir"},
    "yam": {"yam"},
    "livestock": {"cattle", "goat", "sheep", "poultry", "livestock"},
}

CROP_TOPICS = {"maize", "rice", "cowpea", "groundnut", "tomato", "yam", "cassava"}

# FIX: the original list only had words like "shuka"/"planting"/"seed" --
# none of which match how people actually phrase a starter question
# ("How to START FARMING maize", "how do I GROW maize", "BEGIN growing rice").
# Without a match here, wants_production stayed False, so the scorer never
# applied its +1.4 boost toward the correct Planting/Land-Preparation
# section -- retrieval picked whichever section happened to embed closest
# by chance (often Harvest), which then got wrongly excluded by the
# storage filter in src/assistant.py's _filter_chunks.
PRODUCTION_TERMS = {
    "shuka", "shukar", "plant", "planting", "sow", "seed", "iri",
    "spacing", "tazara", "zurfi", "gona", "land preparation",
    "farm", "farming", "start", "begin", "grow", "growing", "growth",
    "cultivate", "cultivation", "guide", "how to",
}
STORAGE_TERMS = {"ajiye", "storage", "store", "harvest", "girbi", "aflatoxin"}


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str):
    return re.findall(r"[^\W\d_']+", text.lower(), flags=re.UNICODE)


def _content_terms(text: str):
    return {token for token in _tokens(text) if len(token) > 2 and token not in STOPWORDS}


def _detect_language(source: str, text: str) -> str:
    lower = f"{source} {text}".lower()
    return "hausa" if "hausa" in lower or any(ch in lower for ch in "ɓɗƙ") else "english"


def _detect_topics(source: str, text: str):
    lower = _norm(f"{source} {text}")
    topics = set()
    for topic, terms in TOPIC_TERMS.items():
        if any(term in lower for term in terms):
            topics.add(topic)
    return topics


def _primary_crop(source: str):
    lower = source.lower()
    for crop in CROP_TOPICS:
        if crop in lower:
            return crop
    return None


def detect_query_topics(text: str):
    return _detect_topics("", text)


def primary_crop_from_source(source: str):
    return _primary_crop(source)


def _clean_document(text: str) -> str:
    cleaned = []
    skip_note = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()
        if line.startswith(">") and ("important note" in lower or "team" in lower):
            skip_note = True
            continue
        if skip_note:
            if not line or line.startswith("## "):
                skip_note = False
            else:
                continue
        if lower.startswith("*sources:") or lower.startswith("*tushen bayani"):
            continue
        if "ai-drafted" in lower or "first draft" in lower:
            continue
        cleaned.append(raw_line)
    return "\n".join(cleaned)


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    sections = []
    current = []
    for line in text.splitlines():
        if line.startswith("## ") and current:
            sections.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("\n".join(current).strip())

    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        start = 0
        while start < len(section):
            end = start + chunk_size
            chunks.append(section[start:end].strip())
            start += chunk_size - overlap

    return [chunk for chunk in chunks if len(chunk) > 40]


def load_corpus():
    docs = []
    seen_text = set()
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "**", "*.*"), recursive=True)):
        if not path.endswith((".md", ".txt")):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = _clean_document(f.read())
        source = os.path.relpath(path, CORPUS_DIR)
        for i, chunk in enumerate(chunk_text(text)):
            fingerprint = hashlib.sha1(_norm(chunk).encode("utf-8")).hexdigest()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            docs.append(
                {
                    "source": source,
                    "chunk_id": i,
                    "text": chunk,
                    "language": _detect_language(source, chunk),
                    "topics": sorted(_detect_topics(source, chunk)),
                    "terms": sorted(_content_terms(chunk)),
                }
            )
    return docs


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = load_corpus()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add .md/.txt files first.")
        return

    print(f"Loaded {len(docs)} unique chunks from corpus. Embedding...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings = model.encode(
        [doc["text"] for doc in docs],
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(META_PATH, "wb") as f:
        pickle.dump({"embed_model": EMBED_MODEL_NAME, "docs": docs}, f)
    print(f"Index built: {len(docs)} chunks -> {INDEX_PATH}")


class Retriever:
    def __init__(self):
        if not os.path.exists(INDEX_PATH) or not os.path.exists(META_PATH):
            raise FileNotFoundError("No index found. Run: python -m src.rag --build")
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            meta = pickle.load(f)
        if isinstance(meta, dict):
            self.docs = meta.get("docs", [])
            self.index_model = meta.get("embed_model")
        else:
            self.docs = meta
            self.index_model = None
        try:
            self.model = SentenceTransformer(EMBED_MODEL_NAME, local_files_only=True)
        except TypeError:
            self.model = SentenceTransformer(EMBED_MODEL_NAME)
        except Exception as exc:
            raise RuntimeError(
                "Embedding model is not available in the local cache. Run "
                "`python -m src.rag --build` once with internet access, then restart AgriSec."
            ) from exc
        if self.index_model != EMBED_MODEL_NAME:
            raise RuntimeError(
                "RAG index was built with an older embedding model. Run: python -m src.rag --build"
            )

    def _raw_search(self, text: str, top_k: int):
        vec = self.model.encode([text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        top_k = min(max(top_k, 1), len(self.docs))
        scores, idxs = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            doc = self.docs[idx]
            results.append({**doc, "score": float(score), "raw_score": float(score)})
        return results

    def query(self, text: str, top_k: int = 3, prefer_hausa: bool = False, min_score: float = 0.28):
        query_terms = _content_terms(text)
        query_topics = _detect_topics("", text)
        lower_query = _norm(text)
        wants_production = any(term in lower_query for term in PRODUCTION_TERMS)
        wants_storage = any(term in lower_query for term in STORAGE_TERMS)
        pool = self._raw_search(text, top_k * 10)
        pool_keys = {(item["source"], item["chunk_id"]) for item in pool}

        for doc in self.docs:
            key = (doc["source"], doc["chunk_id"])
            if key in pool_keys:
                continue
            doc_terms = set(doc.get("terms", []))
            doc_topics = set(doc.get("topics", []))
            topic_overlap = len(query_topics & doc_topics)
            term_overlap = len(query_terms & doc_terms)
            if topic_overlap >= 2 or (prefer_hausa and doc.get("language") == "hausa" and topic_overlap >= 1) or term_overlap >= 3:
                pool.append({**doc, "score": 0.0, "raw_score": 0.0})
                pool_keys.add(key)

        reranked = []
        for item in pool:
            item_terms = set(item.get("terms", []))
            item_topics = set(item.get("topics", []))
            overlap = len(query_terms & item_terms)
            topic_overlap = len(query_topics & item_topics)
            topic_coverage = topic_overlap / len(query_topics) if query_topics else 0.0

            score = item["raw_score"]
            score += min(overlap, 5) * 0.035
            score += topic_overlap * 0.20
            score += topic_coverage * 0.30
            if prefer_hausa and item.get("language") == "hausa":
                score += 0.25
            source_lower = item.get("source", "").lower()
            item_text = _norm(item.get("text", ""))
            requested_crops = query_topics & CROP_TOPICS
            primary_crop = _primary_crop(item.get("source", ""))
            if requested_crops and primary_crop in requested_crops:
                score += 0.55

            # FIX: these "hausa_maize"/"hausa_rice" source-name bonuses used
            # to apply unconditionally whenever the topic matched, regardless
            # of whether the USER wanted a Hausa answer. That let the Hausa
            # doc outrank the correct English doc on a plain English query
            # (exactly what happened with "How to start farming maize").
            # Now gated on prefer_hausa, matching the intent of every other
            # Hausa-specific bonus in this function.
            if prefer_hausa:
                if "maize" in query_topics and "hausa_maize" in source_lower:
                    score += 0.45
                if "rice" in query_topics and "hausa_rice" in source_lower:
                    score += 0.45
            elif item.get("language") == "hausa":
                # Mirror image: when the user did NOT ask for Hausa, actively
                # downrank Hausa-source chunks so they don't win purely on a
                # crop-name match against an English query.
                score -= 0.50

            if "storage" in query_topics and (
                "storage" in source_lower or "post_harvest" in source_lower
            ):
                score += 0.35
            if wants_production and any(term in item_text for term in PRODUCTION_TERMS):
                score += 0.45
            if wants_production and (
                item_text.startswith("## shuka")
                or item_text.startswith("## land preparation")
                or item_text.startswith("## planting")
            ):
                score += 1.4
            if wants_production and (
                item_text.startswith("## gabatarwa")
                or item_text.startswith("## overview")
            ):
                score -= 0.9
            if wants_storage and any(term in item_text for term in STORAGE_TERMS):
                score += 0.45
            if "storage" not in query_topics and (
                "storage" in source_lower or "post_harvest" in source_lower
            ):
                score -= 0.65
            if requested_crops and primary_crop and primary_crop not in requested_crops:
                score -= 0.45
            if query_topics and not topic_overlap:
                score -= 0.18
            if item["raw_score"] < min_score and not topic_overlap:
                continue
            reranked.append({**item, "score": score, "topic_overlap": topic_overlap})

        reranked.sort(key=lambda row: row["score"], reverse=True)

        selected = []
        seen_text = set()
        seen_source = {}
        # FIX: was hard-capped at 1 chunk per source no matter what, even
        # when top_k allowed for more -- meaning every answer (in every log
        # you sent) was built from exactly one chunk of context, regardless
        # of how rich the corpus actually is. If that single chunk happened
        # to be the wrong section, the whole answer failed with nothing else
        # to fall back on. Two chunks per source still avoids one document
        # crowding out everything else, but gives the production-vs-other
        # filtering in src/assistant.py something to actually choose between.
        MAX_PER_SOURCE = 2

        if prefer_hausa:
            hausa_relevant = [
                item for item in reranked
                if item.get("language") == "hausa" and item.get("topic_overlap", 0) > 0
            ]
            if hausa_relevant:
                best_hausa = max(
                    hausa_relevant,
                    key=lambda item: (item.get("topic_overlap", 0), item["score"]),
                )
                selected.append(best_hausa)
                seen_text.add(hashlib.sha1(_norm(best_hausa["text"]).encode("utf-8")).hexdigest())
                seen_source[best_hausa["source"]] = 1

        for item in reranked:
            text_key = hashlib.sha1(_norm(item["text"]).encode("utf-8")).hexdigest()
            if text_key in seen_text:
                continue
            source_count = seen_source.get(item["source"], 0)
            if source_count >= MAX_PER_SOURCE:
                continue
            seen_text.add(text_key)
            seen_source[item["source"]] = source_count + 1
            selected.append(item)
            if len(selected) >= top_k:
                break

        return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--hausa", action="store_true", help="Boost Hausa-source docs for this query")
    args = parser.parse_args()

    if args.build:
        build_index()
    elif args.query:
        r = Retriever()
        for res in r.query(args.query, prefer_hausa=args.hausa):
            print(f"[{res['score']:.3f}] {res['source']} :: {res['text'][:180]}...")
    else:
        parser.print_help()