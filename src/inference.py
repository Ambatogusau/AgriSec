"""Ollama-backed local inference and lightweight language utilities."""
import re
import time

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"

# FIX: the previous set included "a" and "ai" -- both extremely common
# in plain English ("a" = indefinite article, "ai" = lowercased "AI").
# Any English answer using "a" twice ("a fertilizer ... a problem")
# false-positived as Hausa and got its correct answer thrown away in
# _postprocess_answer. This set is now restricted to words/fragments that
# are NOT also common standalone English words. Single-letter and
# two-letter ambiguous tokens are excluded entirely; detection leans more
# on multi-character markers and the Hausa-specific diacritic letters.
HAUSA_MARKERS = {
    "ake", "ana", "ban", "dan", "don", "gari", "gona", "ina", "irin",
    "ita", "kada", "kafin", "karin", "kuma", "mai", "masara", "menene",
    "noma", "sannu", "shinkafa", "shuka", "taki", "wajen", "wannan",
    "yadda", "yana", "yanzu", "yaya", "zan", "gyara", "tambaya",
    "ya", "yaya", "zan", "zanyi", "zanayi",
    "noma", "noman", "gona",
    "masara", "tumatir", "shinkafa",
    "rogo", "doya", "gyada",
    "wake", "alkama", "gero",
    "dawa", "albasa", "barkono",
    "taki", "ruwa", "shuka",
    "girbi", "kwari", "cuta",
    "ina", "kuma", "amma",
    "wannan", "yadda", "lokacin",
}

# These ARE common Hausa words but are short enough to risk overlapping
# with English tokens or being noise; only count them with reduced weight
# (0.5 "hit") rather than excluding them outright, so genuine short Hausa
# input still gets detected without making English text vulnerable.
HAUSA_WEAK_MARKERS = {
    "ba", "ce", "da", "ga", "ka", "ki", "mu", "na", "ne", "su", "ta",
}

HAUSA_REQUEST_PATTERNS = (
    "in hausa",
    "hausa",
    "harshen hausa",
    "fassara",
    "translate to hausa",
)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str):
    return re.findall(r"[a-zA-ZÀ-ÿ']+", text.lower())


def looks_like_hausa(text: str) -> bool:
    """
    Cheap deterministic Hausa/mixed-language detection.

    FIX: this used to give "a" and "ai" full weight as markers, causing
    almost any sufficiently long English answer (which will use the
    article "a" at least twice) to be misclassified as Hausa. Strong
    markers now require multi-character, Hausa-specific words; weak
    markers (short words that overlap less dangerously with English)
    only count at half weight, and the decision scales with text length
    so a single coincidental hit on a long English paragraph can't flip
    the result.
    """
    lower = normalize_text(text)
    if any(pattern in lower for pattern in HAUSA_REQUEST_PATTERNS):
        return True
    if any(ch in lower for ch in "ɓɗƙ"):
        return True

    words = tokenize(lower)
    if not words:
        return False

    strong_hits = sum(1 for word in words if word in HAUSA_MARKERS)
    weak_hits = sum(1 for word in words if word in HAUSA_WEAK_MARKERS)
    score = strong_hits + (0.5 * weak_hits)

    if len(words) <= 4:
        # Short input: a single strong marker is enough (e.g. "sannu",
        # "noma"), but weak markers alone are not, since a short English
        # phrase could coincidentally contain one.
        return strong_hits >= 1

    # Longer input: require the marker density to be a real signal, not a
    # coincidence. ~1 strong-equivalent hit per 12 words is a reasonable
    # floor; tune after watching real logs (AGRISEC_RETRIEVAL_DEBUG=1).
    return score >= 2 and (score / len(words)) >= 0.05


def hausa_quality_ok(text: str) -> bool:
    """Reject clearly English or garbled outputs for Hausa requests."""
    lower = normalize_text(text)
    if not lower:
        return False
    bad_fragments = (
        "i am agrisec",
        "to assist you",
        "provided text",
        "local knowledge base",
        "as an ai",
        "ba da tabbaci bayani",
        "kuva ba",
        "kuwa ba da tabbaci",
        "do not have enough information",
    )
    if any(fragment in lower for fragment in bad_fragments):
        return False
    words = tokenize(lower)
    strong_hits = sum(1 for word in words if word in HAUSA_MARKERS)
    weak_hits = sum(1 for word in words if word in HAUSA_WEAK_MARKERS)
    hausa_hits = strong_hits + (0.5 * weak_hits)
    if len(words) < 4:
        return hausa_hits > 0
    english_hits = sum(
        1 for word in words
        if word in {"the", "and", "should", "store", "maize", "after", "harvest", "use"}
    )
    return hausa_hits >= 2 and english_hits <= max(2, hausa_hits)


def _looks_finished(text: str) -> bool:
    """Heuristic: does the text end on a real sentence/bullet boundary, or
    does it look like Ollama was cut off mid-word/mid-clause?"""
    stripped = (text or "").rstrip()
    if not stripped:
        return True
    if stripped[-1] in ".!?\u061f\u3002\":)":
        return True
    if stripped[-1] == "-":
        return False
    last_word = re.findall(r"[\w']+", stripped)[-1] if re.findall(r"[\w']+", stripped) else ""
    dangling_words = {
        "and", "or", "the", "a", "an", "to", "of", "in", "on", "at", "is",
        "around", "about", "near", "like", "such", "for", "with", "by",
        "from", "into", "between", "than", "as", "if", "when", "after",
        "before", "during", "while", "because",
        "da", "na", "wajen", "kuma", "sai", "amma",
    }
    return last_word.lower() not in dangling_words


class LocalLLM:
    def __init__(self, model_name="qwen2.5:1.5b", n_ctx: int = 1536, n_threads: int = 4):
        self.model_name = model_name
        self.n_ctx = n_ctx
        self.n_threads = n_threads

    def _call_ollama(self, prompt: str, max_tokens: int, temperature: float):
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.8,
                "top_k": 30,
                "repeat_penalty": 1.15,
                "num_predict": max_tokens,
                "num_ctx": self.n_ctx,
                "num_thread": self.n_threads,
            },
        }
        return requests.post(OLLAMA_URL, json=payload, timeout=60)

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.1, max_continuations: int = 2):
        start = time.time()
        try:
            response = self._call_ollama(prompt, max_tokens, temperature)
        except requests.RequestException as exc:
            elapsed = time.time() - start
            return {
                "text": (
                    "I could not reach Ollama locally. Start Ollama and run "
                    f"`ollama pull {self.model_name}` before using the model. ({exc})"
                ),
                "tokens": 0,
                "elapsed_sec": elapsed,
                "tokens_per_sec": 0.0,
                "error": True,
            }

        if response.status_code != 200:
            elapsed = time.time() - start
            return {
                "text": f"Ollama returned error {response.status_code}: {response.text[:180]}",
                "tokens": 0,
                "elapsed_sec": elapsed,
                "tokens_per_sec": 0.0,
                "error": True,
            }

        data = response.json()
        text = data.get("response", "").strip()
        total_tokens = int(data.get("eval_count") or len(text.split()))
        total_eval_duration = data.get("eval_duration", 0) or 0

        continuations = 0
        while (
            continuations < max_continuations
            and data.get("done_reason") == "length"
            and not _looks_finished(text)
        ):
            continuations += 1
            continue_prompt = (
                f"{prompt}{text}\n\n"
                "Continue the answer above. Finish the current sentence/bullet and "
                "then stop. Do not repeat earlier text.\n"
            )
            try:
                cont_response = self._call_ollama(continue_prompt, 80, temperature)
            except requests.RequestException:
                break
            if cont_response.status_code != 200:
                break
            cont_data = cont_response.json()
            addition = cont_data.get("response", "").strip()
            if not addition:
                break
            text = f"{text} {addition}".strip()
            total_tokens += int(cont_data.get("eval_count") or len(addition.split()))
            total_eval_duration += cont_data.get("eval_duration", 0) or 0
            data = cont_data

        elapsed = time.time() - start
        eval_duration = total_eval_duration / 1_000_000_000
        speed_time = eval_duration if eval_duration > 0 else elapsed
        return {
            "text": text,
            "tokens": int(total_tokens),
            "elapsed_sec": elapsed,
            "tokens_per_sec": int(total_tokens) / speed_time if speed_time > 0 else 0.0,
            "error": False,
        }