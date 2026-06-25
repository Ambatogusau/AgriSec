"""
Local Retrieval-Augmented Generation over the agricultural knowledge base.
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

# FIX: "millet" and "sorghum" were completely missing from this dict before.
# Any question about millet/sorghum got NO topic match at all (confirmed by
# "Detected Topic: []" in the debug log), so none of the crop-matching or
# crop-penalizing logic below had anything to act on -- the doc lost purely
# on raw embedding similarity to whatever crop happened to be closest
# (maize, groundnut). Also added onion, pepper, cotton, sesame, cowpea
# variants and aquaculture/fish so the same silent gap doesn't repeat for
# every other crop added since this dict was last touched.
TOPIC_TERMS = {
    "maize": {"maize", "corn", "masara"},
    "rice": {"rice", "shinkafa"},
    "cassava": {"cassava", "rogo"},
    "millet_sorghum": {"millet", "sorghum", "gero", "dawa"},
    "storage": {"store", "storage", "stored", "harvest", "post", "aflatoxin", "dry", "drying", "ajiye", "girbi", "bushe"},
    "fall_armyworm": {"fall", "armyworm", "tsutsa"},
    "fertilizer": {"fertilizer", "fertiliser", "taki", "npk", "urea"},
    "cowpea": {"cowpea", "wake"},
    "groundnut": {"groundnut", "peanut", "gyada"},
    "tomato": {"tomato", "tumatir"},
    "yam": {"yam", "doya"},
    "onion_pepper": {"onion", "pepper", "albasa", "tattasai"},
    "cotton_sesame": {"cotton", "sesame", "auduga", "ridi"},
    "aquaculture": {"fish", "catfish", "tilapia", "kifi"},
    "wheat": {"wheat", "alkama"},
    "sweet_potato": {"sweet potato", "sweetpotato", "dankali", "dankalin turawa"},
    "oil_palm": {"oil palm", "palm oil", "dabino"},
    "livestock": {"cattle", "goat", "sheep", "poultry", "livestock"},
}

# FIX: same gap as above -- this set drives both the "+0.55 matches
# requested crop" bonus AND the "-0.45 wrong crop" penalty. Without
# "millet_sorghum" here, a millet question could never get either signal.
# Added wheat/sweet_potato/oil_palm now too, so the three crops added in
# this session don't repeat the exact same silent gap.
CROP_TOPICS = {
    "maize", "rice", "cowpea", "groundnut", "tomato", "yam", "cassava",
    "millet_sorghum", "onion_pepper", "cotton_sesame",
    "wheat", "sweet_potato", "oil_palm",
}

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
    """
    FIX: previously looped over CROP_TOPICS (a set, so iteration order is not
    guaranteed) checking `if crop in lower`. Since "millet_sorghum_production.md"
    contains both potential crop substrings once millet/sorghum existed, this
    is resolved with an explicit, deterministic check list instead of relying
    on set-membership substring matching, which is also safer against future
    filename collisions (e.g. "cotton_sesame_production.md" matching two crops).
    """
    lower = source.lower()
    explicit = [
        ("millet_sorghum", "millet_sorghum"),
        ("onion_pepper", "onion_pepper"),
        ("cotton_sesame", "cotton_sesame"),
        ("sweet_potato", "sweet_potato"),
        ("oil_palm", "oil_palm"),
        ("wheat", "wheat"),
        ("maize", "maize"),
        ("rice", "rice"),
        ("cowpea", "cowpea"),
        ("groundnut", "groundnut"),
        ("tomato", "tomato"),
        ("yam", "yam"),
        ("cassava", "cassava"),
    ]
    for needle, crop in explicit:
        if needle in lower:
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

            if prefer_hausa:
                if "maize" in query_topics and "hausa_maize" in source_lower:
                    score += 0.45
                if "rice" in query_topics and "hausa_rice" in source_lower:
                    score += 0.45
            elif item.get("language") == "hausa":
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
            # FIX: this is the line that actually does crop exclusion -- it was
            # always correct in logic, it just never FIRED for millet/sorghum
            # because requested_crops was always empty for those questions
            # (see TOPIC_TERMS/CROP_TOPICS fix above). No change needed here,
            # just confirming this penalty now actually reaches maize/groundnut
            # when the question is about millet.
            if requested_crops and primary_crop and primary_crop not in requested_crops:
                score -= 0.65  # strengthened from 0.45 -> 0.65 for a harder exclusion
            if query_topics and not topic_overlap:
                score -= 0.18
            if item["raw_score"] < min_score and not topic_overlap:
                continue
            reranked.append({**item, "score": score, "topic_overlap": topic_overlap})

        reranked.sort(key=lambda row: row["score"], reverse=True)

        selected = []
        seen_text = set()
        seen_source = {}
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
