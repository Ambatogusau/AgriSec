"""
AgriSec section-aware Retrieval-Augmented Generation engine.

ARCHITECTURE (see class docstrings for detail):

    SectionParser     -> splits a raw markdown document into (heading, body)
                         sections at ANY heading level, with a paragraph-split
                         fallback for documents with no headings at all.

    MetadataExtractor  -> for each section, determines its topic (crop or
                          cross-cutting subject), its SECTION TAG (planting /
                          fertilizer / pest_disease / weeding / harvest_storage
                          / general), and its language.

    ChunkBuilder        -> turns parsed sections into final chunk dicts,
                          sub-splitting any section that is still too long,
                          and deduplicating identical text across documents.

    IntentDetector      -> the query-side mirror of MetadataExtractor: given a
                          user question, determines requested topic(s) and
                          requested section(s) (NOT language -- language
                          detection for replies stays in src/inference.py,
                          this module only consumes the `prefer_hausa` flag
                          callers already pass in).

    Reranker            -> scores a candidate chunk against a detected intent.

    Retriever           -> orchestrates FAISS search + inverted-index lookups
                          + hard topic/section/language filtering + Reranker
                          scoring + deduplication. This is the only class
                          other modules touch directly.

COMPATIBILITY: assistant.py and web_app.py only ever import
`CROP_TOPICS`, `Retriever`, `detect_query_topics`, `primary_crop_from_source`
from this module. All four behave the same as before from the caller's
point of view; everything else here is new internal structure.

Build the index:
    python -m src.rag --build

Query (debugging):
    python -m src.rag --query "Wane taki ya dace da shinkafa?" --hausa
"""
import argparse
import glob
import hashlib
import os
import pickle
import re
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "corpus")
INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "index")
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.pkl")

EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# A section is split further only if it's still this long after heading-level
# splitting -- most real sections in this corpus (a paragraph or a short list
# under one "## Heading") are well under this, so in practice almost every
# chunk now corresponds to exactly one heading, which is the whole point.
MAX_SECTION_CHARS = 900
PARAGRAPH_OVERLAP = 0  # paragraph-level sub-splits don't need char overlap

# ---------------------------------------------------------------------------
# Topic vocabulary (crop + cross-cutting subjects). Unchanged in *purpose*
# from the previous version -- this still drives detect_query_topics() and
# the legacy `topics` field every chunk carries, which assistant.py's own
# filtering logic depends on. Extend this dict whenever a new crop/subject
# is added to the corpus; it is the single most common source of silent bugs
# in this system (see the millet/sorghum incident).
# ---------------------------------------------------------------------------
TOPIC_KEYWORDS = {
    "maize": {"maize", "corn", "masara"},
    "rice": {"rice", "shinkafa"},
    "cassava": {"cassava", "rogo"},
    "millet_sorghum": {"millet", "sorghum", "gero", "dawa"},
    "fall_armyworm": {"fall", "armyworm", "tsutsa"},
    "fertilizer": {"fertilizer", "fertiliser", "taki", "npk", "urea"},
    "cowpea": {"cowpea", "wake"},
    "groundnut": {"groundnut", "peanut", "gyada"},
    "tomato": {"tomato", "tumat"},  # "tumat" is a deliberate stem, not a full
                                      # word -- matches "tumatir", "tumatur",
                                      # "tumatuna" and other common spelling
                                      # variants instead of requiring an exact
                                      # match (this was the cause of a real
                                      # missed-detection bug previously).
    "yam": {"yam", "doya"},
    "onion_pepper": {"onion", "pepper", "albasa", "tattasai"},
    "cotton_sesame": {"cotton", "sesame", "auduga", "ridi"},
    "aquaculture": {"fish", "catfish", "tilapia", "kifi"},
    "wheat": {"wheat", "alkama"},
    "sweet_potato": {"sweet potato", "sweetpotato", "dankali"},
    "oil_palm": {"oil palm", "palm oil", "dabino"},
    "livestock": {"cattle", "goat", "sheep", "poultry", "livestock"},
    "storage": {"store", "storage", "stored", "harvest", "post", "aflatoxin",
                "dry", "drying", "ajiye", "girbi", "bushe"},
}

CROP_TOPICS = {
    "maize", "rice", "cowpea", "groundnut", "tomato", "yam", "cassava",
    "millet_sorghum", "onion_pepper", "cotton_sesame",
    "wheat", "sweet_potato", "oil_palm",
}

# Deterministic filename -> crop lookup. Order matters: compound names
# ("millet_sorghum", "cotton_sesame", "sweet_potato", "oil_palm") are checked
# before any single crop name that might be a substring of another filename.
_PRIMARY_CROP_LOOKUP = (
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
)

# ---------------------------------------------------------------------------
# NEW: section vocabulary. This is the core addition that fixes "whole
# document returned" -- every chunk gets tagged with which part of a guide
# it actually is, and every query gets tagged with which part the user is
# actually asking about, and retrieval is hard-filtered on the overlap.
# ---------------------------------------------------------------------------
SECTION_TAGS = (
    "overview", "planting", "fertilizer", "pest_disease",
    "weeding", "harvest_storage", "general",
)

# Used by MetadataExtractor against a SECTION HEADING (e.g. "## Fertilizer",
# "## Taki (Fertilizer)", "## Manyan Kwari da Cututtuka"). Checked in this
# order; first match wins per heading. A heading that matches nothing here
# is tagged "general" rather than guessed at.
_HEADING_SECTION_RULES = (
    ("fertilizer", {"fertilizer", "npk", "urea", "manure", "taki"}),
    ("harvest_storage", {"harvest", "post-harvest", "post harvest", "storage",
                          "aflatoxin", "girbi", "drying", "dry", "ajiye"}),
    ("pest_disease", {"pest", "insect", "disease", "virus", "cuta",
                       "kwari", "cututtuka", "blight", "rot", "mosaic",
                       "aphid", "armyworm", "borer", "locust", "weevil",
                       "vaccination", "vaccine"}),
    ("weeding", {"weed", "striga", "ciyawa"}),
    ("planting", {"planting", "land preparation", "spacing", "nursery",
                  "staking", "seedling", "shuka", "dasawa", "irin",
                  "varieties", "variety", "sowing", "transplant"}),
    ("overview", {"overview", "gabatarwa", "background", "why", "introduction"}),
)

# Used by IntentDetector against the USER'S QUESTION. Deliberately a
# slightly different (broader) vocabulary than the heading rules above,
# because people phrase questions very differently from how documents are
# titled -- e.g. nobody titles a section "## How do I start", but plenty of
# farmers ask "how do I start farming maize" and mean the planting section.
_QUERY_SECTION_RULES = (
    ("fertilizer", {"taki", "fertilizer", "fertiliser", "npk", "urea", "manure"}),
    ("harvest_storage", {"girbi", "harvest", "ajiye", "store", "storage",
                          "aflatoxin", "dry", "drying"}),
    ("pest_disease", {"kwari", "pest", "disease", "cuta", "armyworm",
                       "aphid", "borer", "locust", "virus", "insect",
                       "weevil", "vaccine", "vaccination"}),
    ("weeding", {"ciyawa", "weed", "striga"}),
    ("planting", {"shuka", "shukar", "plant", "planting", "sow", "seed",
                  "iri", "spacing", "tazara", "zurfi", "farm", "farming",
                  "start", "begin", "grow", "growing", "growth",
                  "cultivate", "cultivation", "how to", "noma", "noman"}),
)

STOPWORDS = {
    "about", "after", "again", "also", "anything", "before", "could", "does",
    "explain", "from", "give", "have", "help", "into", "more", "need", "please",
    "safe", "safely", "should", "tell", "that", "their", "there", "this", "what",
    "when", "where", "which", "with", "would", "your", "bayani", "karin", "kuma",
    "yana", "ina", "yadda", "wajen", "don", "da", "na", "a", "zan",
}


# ---------------------------------------------------------------------------
# Small text utilities shared across classes
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _tokens(text: str):
    return re.findall(r"[^\W\d_']+", text.lower(), flags=re.UNICODE)


def _content_terms(text: str):
    return {token for token in _tokens(text) if len(token) > 2 and token not in STOPWORDS}


def _detect_language(source: str, text: str) -> str:
    lower = f"{source} {text}".lower()
    return "hausa" if "hausa" in lower or any(ch in lower for ch in "ɓɗƙ") else "english"


def _match_keywords(lower_text: str, rules):
    """Shared helper: walk an ordered (tag, keyword_set) rule list and return
    the set of ALL matching tags (not just the first) -- a heading or query
    can legitimately touch more than one section, e.g. "## Weeding and
    Disease Management" is genuinely both weeding and pest_disease."""
    matched = set()
    for tag, keywords in rules:
        if any(kw in lower_text for kw in keywords):
            matched.add(tag)
    return matched


def _detect_topics(source: str, text: str):
    lower = _norm(f"{source} {text}")
    topics = set()
    for topic, terms in TOPIC_KEYWORDS.items():
        if any(term in lower for term in terms):
            topics.add(topic)
    return topics


def _primary_crop(source: str):
    lower = source.lower()
    for needle, crop in _PRIMARY_CROP_LOOKUP:
        if needle in lower:
            return crop
    return None


# Public, stable API (used by assistant.py / web_app.py) -------------------
def detect_query_topics(text: str):
    return _detect_topics("", text)


def primary_crop_from_source(source: str):
    return _primary_crop(source)


def detect_query_sections(text: str):
    """NEW public helper. Not used by the current assistant.py, but exposed
    so a future small enhancement there (e.g. carrying the previous crop
    across a section-shifting follow-up like "explain fertilizer" after
    "how do I grow rice") has something to call without touching rag.py
    again. Returns a set of SECTION_TAGS."""
    return _match_keywords(_norm(text), _QUERY_SECTION_RULES)


# ---------------------------------------------------------------------------
# SectionParser
# ---------------------------------------------------------------------------
class SectionParser:
    """
    Splits a cleaned markdown document into (heading, body) pairs at ANY
    heading level (#, ##, ###, ...), not just "## ". This matters because a
    few corpus documents use "### " for sub-points under a "## " section,
    and the old parser silently merged those into their parent section,
    sometimes producing a single oversized chunk that mixed two real topics
    (e.g. "## Pests and Diseases" containing both "### Aphids" and
    "### Leaf Spot" as one chunk).

    If a document has NO headings at all (rare, but the parser must not
    crash or silently return nothing), falls back to splitting on blank
    lines (paragraphs).
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")

    @classmethod
    def parse(cls, text: str):
        """Returns a list of dicts: {"heading": str, "body": str, "level": int}.
        `heading` is "" for the paragraph-fallback case."""
        lines = text.splitlines()
        sections = []
        current_heading = ""
        current_level = 0
        current_body = []

        def flush():
            body = "\n".join(current_body).strip()
            if current_heading or body:
                sections.append({
                    "heading": current_heading,
                    "body": body,
                    "level": current_level,
                })

        found_any_heading = False
        for line in lines:
            match = cls.HEADING_RE.match(line)
            if match:
                found_any_heading = True
                flush()
                current_level = len(match.group(1))
                current_heading = match.group(2).strip()
                current_body = []
            else:
                current_body.append(line)
        flush()

        if found_any_heading:
            # Drop the bare document title (level-1 "# Title" with no body
            # of its own beyond the heading text) -- it carries no
            # retrievable content on its own and would otherwise show up as
            # a near-empty, untagged "general" chunk.
            sections = [s for s in sections if not (s["level"] == 1 and len(s["body"]) < 20)]
            return sections

        # Fallback: no headings anywhere in this document. Split on blank
        # lines into paragraph chunks instead of returning one giant blob.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        return [{"heading": "", "body": p, "level": 0} for p in paragraphs]


# ---------------------------------------------------------------------------
# MetadataExtractor
# ---------------------------------------------------------------------------
class MetadataExtractor:
    """Given a document's source path and one parsed section, determines the
    topic tags, section tags, and language for that section."""

    @staticmethod
    def section_tags_for_heading(heading: str, body: str):
        lower_heading = _norm(heading)
        tags = _match_keywords(lower_heading, _HEADING_SECTION_RULES)
        if tags:
            return tags
        # Heading itself didn't match anything specific (e.g. "## Notes",
        # or a Hausa heading whose translated half is in parentheses and
        # got missed) -- fall back to scanning the body text once before
        # giving up and tagging "general". This recovers cases like a
        # heading "## Gabatarwa (Overview)" where "overview" is present but
        # wrapped in parentheses -- _norm already lowercases so this should
        # already match above; the body-scan fallback mainly helps headings
        # with no recognizable English/Hausa keyword at all.
        lower_body = _norm(body[:300])  # only scan the opening of the body
        tags = _match_keywords(lower_body, _HEADING_SECTION_RULES)
        return tags or {"general"}

    @staticmethod
    def topics_for_chunk(source: str, heading: str, body: str):
        return _detect_topics(source, f"{heading} {body}")

    @staticmethod
    def language_for_chunk(source: str, heading: str, body: str):
        return _detect_language(source, f"{heading} {body}")


# ---------------------------------------------------------------------------
# Document cleaning (item 13: strip developer/translation notes before
# anything reaches FAISS)
# ---------------------------------------------------------------------------
def _clean_document(text: str) -> str:
    """
    Strips anything that is metadata-about-the-document rather than
    agricultural knowledge itself:
      - blockquote lines (">...") -- every team/AI-translation disclaimer in
        this corpus uses blockquote style, so this is now a blanket rule
        instead of a keyword-matched one (more robust to new disclaimer
        wording we haven't seen yet).
      - HTML comments <!-- ... -->
      - lines starting with TODO / NOTE: / FIXME (case-insensitive)
      - "*Sources:" / "*Tushen bayani" citation footers
    """
    cleaned = []
    in_html_comment = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        lower = line.lower()

        if in_html_comment:
            if "-->" in line:
                in_html_comment = False
            continue
        if line.startswith("<!--"):
            in_html_comment = "-->" not in line
            continue
        if line.startswith(">"):
            continue  # blanket blockquote strip -- see docstring
        if lower.startswith(("todo", "note:", "fixme")):
            continue
        if lower.startswith("*sources:") or lower.startswith("*tushen bayani"):
            continue
        if "ai-drafted" in lower or "first draft" in lower:
            continue

        cleaned.append(raw_line)
    return "\n".join(cleaned)


# ---------------------------------------------------------------------------
# ChunkBuilder
# ---------------------------------------------------------------------------
class ChunkBuilder:
    """Turns a cleaned document into final chunk dicts ready for embedding."""

    @staticmethod
    def _split_long_body(heading: str, body: str):
        """If a section is still long after heading-level splitting, break
        it into paragraph-sized sub-chunks, each re-prefixed with the same
        heading so retrieval context stays intelligible on its own."""
        if len(body) <= MAX_SECTION_CHARS:
            return [body]
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        if len(paragraphs) <= 1:
            # No paragraph breaks to exploit -- fall back to a hard character
            # split as a last resort rather than passing one huge chunk on.
            chunks = []
            start = 0
            while start < len(body):
                chunks.append(body[start:start + MAX_SECTION_CHARS].strip())
                start += MAX_SECTION_CHARS
            return chunks

        grouped, current = [], ""
        for para in paragraphs:
            candidate = f"{current}\n\n{para}".strip() if current else para
            if len(candidate) > MAX_SECTION_CHARS and current:
                grouped.append(current)
                current = para
            else:
                current = candidate
        if current:
            grouped.append(current)
        return grouped

    @classmethod
    def build_for_document(cls, source: str, raw_text: str):
        cleaned = _clean_document(raw_text)
        sections = SectionParser.parse(cleaned)

        chunks = []
        chunk_id = 0
        for section in sections:
            heading = section["heading"]
            body = section["body"]
            if not body or len(body) < 20:
                continue

            section_tags = MetadataExtractor.section_tags_for_heading(heading, body)
            language = MetadataExtractor.language_for_chunk(source, heading, body)

            for piece in cls._split_long_body(heading, body):
                if len(piece) < 20:
                    continue
                full_text = f"## {heading}\n{piece}".strip() if heading else piece
                topics = MetadataExtractor.topics_for_chunk(source, heading, piece)
                chunks.append({
                    "source": source,
                    "chunk_id": chunk_id,
                    "text": full_text,
                    "heading": heading,
                    "language": language,
                    "topics": sorted(topics),
                    "sections": sorted(section_tags),
                    "terms": sorted(_content_terms(piece)),
                })
                chunk_id += 1
        return chunks


# Backward-compatible module-level name some external tooling/tests might
# still reference; delegates entirely to the new builder.
def chunk_text(text: str, source: str = ""):
    return [c["text"] for c in ChunkBuilder.build_for_document(source, text)]


def load_corpus():
    """Walks the corpus directory, builds section-level chunks for every
    document, and removes exact-duplicate chunk text across files (item 10)."""
    docs = []
    seen_text = set()
    for path in sorted(glob.glob(os.path.join(CORPUS_DIR, "**", "*.*"), recursive=True)):
        if not path.endswith((".md", ".txt")):
            continue
        with open(path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        source = os.path.relpath(path, CORPUS_DIR)
        for chunk in ChunkBuilder.build_for_document(source, raw_text):
            fingerprint = hashlib.sha1(_norm(chunk["text"]).encode("utf-8")).hexdigest()
            if fingerprint in seen_text:
                continue
            seen_text.add(fingerprint)
            docs.append(chunk)
    return docs


def build_index():
    os.makedirs(INDEX_DIR, exist_ok=True)
    docs = load_corpus()
    if not docs:
        print(f"No documents found in {CORPUS_DIR}. Add .md/.txt files first.")
        return

    print(f"Loaded {len(docs)} unique section-level chunks from corpus. Embedding...")
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

    # Quick section-tag distribution printout -- useful for sanity-checking
    # a rebuild without writing a separate debug script.
    from collections import Counter
    section_counts = Counter(tag for doc in docs for tag in doc["sections"])
    print(f"Index built: {len(docs)} chunks -> {INDEX_PATH}")
    print(f"Section tag distribution: {dict(section_counts)}")


# ---------------------------------------------------------------------------
# IntentDetector (query-side)
# ---------------------------------------------------------------------------
class IntentDetector:
    """Determines what a query is actually asking for: which crop/topic, and
    which section of a guide. Language is intentionally NOT detected here --
    callers (assistant.py) already determine reply language via
    src/inference.py's looks_like_hausa() and pass it in as `prefer_hausa`;
    duplicating that logic here would be a second place for it to drift out
    of sync, which is exactly the kind of bug this whole refactor is trying
    to eliminate."""

    @staticmethod
    def detect(text: str):
        topics = detect_query_topics(text)
        sections = detect_query_sections(text)
        crops = topics & CROP_TOPICS
        return {
            "topics": topics,
            "sections": sections,
            "crops": crops,
        }


# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------
class Reranker:
    """Pure scoring function: given a candidate chunk, its raw embedding
    score, the detected query intent, and the prefer_hausa flag, returns a
    final score. Hard filtering (excluding wrong crop/section/language
    entirely) happens in Retriever BEFORE this runs -- Reranker only ranks
    within whatever pool already survived those hard filters."""

    @staticmethod
    def score(chunk: dict, raw_score: float, query_terms: set, intent: dict, prefer_hausa: bool) -> float:
        item_terms = set(chunk.get("terms", []))
        item_topics = set(chunk.get("topics", []))
        item_sections = set(chunk.get("sections", []))

        term_overlap = len(query_terms & item_terms)
        topic_overlap = len(intent["topics"] & item_topics)
        section_overlap = len(intent["sections"] & item_sections)

        score = raw_score
        score += min(term_overlap, 5) * 0.035
        score += topic_overlap * 0.20
        score += section_overlap * 0.50  # section match is a strong signal
                                            # now that it's meaningful -- it's
                                            # the difference between "right
                                            # part of the right guide" and
                                            # "right guide, wrong part".

        primary_crop = primary_crop_from_source(chunk.get("source", ""))
        if intent["crops"] and primary_crop in intent["crops"]:
            score += 0.55

        if prefer_hausa and chunk.get("language") == "hausa":
            score += 0.30

        return score


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------
class Retriever:
    """
    Orchestrates: FAISS similarity search -> inverted-index candidate
    expansion -> hard language/crop/section filtering -> Reranker scoring ->
    deduplication -> top_k selection.

    Hard filtering (new in this version) means a maize question CANNOT
    return a groundnut chunk, and a fertilizer question CANNOT return a
    storage chunk, regardless of how that chunk happened to embed -- this
    was previously only a score penalty, which a high enough raw similarity
    score could still override.
    """

    MAX_PER_SOURCE = 2

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

        # --- Speed (item 14): inverted indices built ONCE at startup, not
        # recomputed per query. Replaces the old approach of looping over
        # every single chunk in the corpus on every query call to find
        # lexical/topic-overlap candidates -- with several hundred
        # section-level chunks now (finer chunking means more of them than
        # the old document-level version), that loop was real, avoidable
        # CPU cost on every turn on hardware that has none to spare.
        self._topic_index = {}
        self._section_index = {}
        self._language_index = {"english": [], "hausa": []}
        for idx, doc in enumerate(self.docs):
            for topic in doc.get("topics", []):
                self._topic_index.setdefault(topic, []).append(idx)
            for section in doc.get("sections", []):
                self._section_index.setdefault(section, []).append(idx)
            self._language_index[doc.get("language", "english")].append(idx)

    # -- internal helpers ---------------------------------------------------
    def _embed_search(self, text: str, top_k: int):
        vec = self.model.encode([text], normalize_embeddings=True)
        vec = np.asarray(vec, dtype="float32")
        top_k = min(max(top_k, 1), len(self.docs))
        scores, idxs = self.index.search(vec, top_k)
        results = {}
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results[idx] = float(score)
        return results

    def _candidate_pool(self, text: str, intent: dict, top_k: int):
        """Single FAISS call for the embedding-similarity pool, plus a fast
        inverted-index lookup (no full corpus scan) for any chunk that
        shares a requested topic/section but didn't surface via raw
        embedding similarity alone -- this is what lets a short, plainly-
        worded Hausa query still find the right section even when the
        multilingual embedding distance to it is mediocre."""
        embed_scores = self._embed_search(text, top_k * 8)
        candidate_idxs = set(embed_scores.keys())

        for topic in intent["topics"]:
            candidate_idxs.update(self._topic_index.get(topic, []))
        for section in intent["sections"]:
            candidate_idxs.update(self._section_index.get(section, []))

        pool = []
        for idx in candidate_idxs:
            doc = self.docs[idx]
            pool.append({**doc, "raw_score": embed_scores.get(idx, 0.0)})
        return pool

    def _apply_hard_filters(self, pool, intent: dict, prefer_hausa: bool):
        """Item 11 (Hausa-first) + crop/section hard exclusion. Each filter
        only narrows the pool if doing so wouldn't eliminate everything --
        every filter here has an explicit, commented fallback so a narrow
        corpus gap degrades gracefully instead of returning nothing."""

        # --- Language: Hausa-first, hard fallback to English -------------
        if prefer_hausa:
            hausa_only = [c for c in pool if c.get("language") == "hausa"]
            if hausa_only:
                pool = hausa_only
            # else: no Hausa source exists for this at all -- fall through
            # to the full (English) pool rather than returning nothing.
        else:
            english_only = [c for c in pool if c.get("language") != "hausa"]
            if english_only:
                pool = english_only
            # else: corpus has ONLY a Hausa source for this topic -- better
            # to surface it (the LLM prompt is English-only downstream, so
            # this is a degraded-but-honest result) than return nothing.

        # --- Crop: hard exclusion of mismatched crops --------------------
        if intent["crops"]:
            crop_matched = [
                c for c in pool
                if primary_crop_from_source(c.get("source", "")) is None
                or primary_crop_from_source(c.get("source", "")) in intent["crops"]
            ]
            if crop_matched:
                pool = crop_matched
            # else: nothing in the corpus for this crop at all -- leave pool
            # as-is rather than emptying it; the relevance-score threshold in
            # assistant.py will catch a genuinely poor match downstream.

        # --- Section: hard exclusion of mismatched sections ---------------
        if intent["sections"]:
            section_matched = [
                c for c in pool
                if set(c.get("sections", [])) & intent["sections"]
            ]
            if section_matched:
                pool = section_matched
            # else: the requested section doesn't exist for this topic in
            # the corpus -- again, leave pool as-is and let the score
            # threshold downstream decide rather than returning nothing.

        return pool

    def _dedupe_and_select(self, ranked, top_k: int):
        selected, seen_text, seen_source = [], set(), {}
        for item in ranked:
            text_key = hashlib.sha1(_norm(item["text"]).encode("utf-8")).hexdigest()
            if text_key in seen_text:
                continue
            source_count = seen_source.get(item["source"], 0)
            if source_count >= self.MAX_PER_SOURCE:
                continue
            seen_text.add(text_key)
            seen_source[item["source"]] = source_count + 1
            selected.append(item)
            if len(selected) >= top_k:
                break
        return selected

    # -- public API (unchanged signature) -----------------------------------
    def query(self, text: str, top_k: int = 3, prefer_hausa: bool = False, min_score: float = 0.28):
        intent = IntentDetector.detect(text)
        query_terms = _content_terms(text)

        pool = self._candidate_pool(text, intent, top_k)
        pool = self._apply_hard_filters(pool, intent, prefer_hausa)

        ranked = []
        for item in pool:
            if item["raw_score"] < min_score and not (intent["topics"] & set(item.get("topics", []))):
                continue
            final_score = Reranker.score(item, item["raw_score"], query_terms, intent, prefer_hausa)
            ranked.append({**item, "score": final_score})

        ranked.sort(key=lambda row: row["score"], reverse=True)
        return self._dedupe_and_select(ranked, top_k)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--hausa", action="store_true", help="Prefer Hausa-source docs for this query")
    args = parser.parse_args()

    if args.build:
        t0 = time.time()
        build_index()
        print(f"Build completed in {time.time() - t0:.1f}s")
    elif args.query:
        r = Retriever()
        intent = IntentDetector.detect(args.query)
        print(f"Detected topics: {sorted(intent['topics'])}")
        print(f"Detected sections: {sorted(intent['sections'])}")
        print(f"Detected crops: {sorted(intent['crops'])}")
        for res in r.query(args.query, prefer_hausa=args.hausa):
            print(f"[{res['score']:.3f}] {res['source']} | sections={res['sections']} "
                  f"| lang={res['language']} :: {res['text'][:160]}...")
    else:
        parser.print_help()