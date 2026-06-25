"""
AgriSec Local Assistant - production RAG orchestration.

Usage:
    python -m src.assistant --model qwen2.5:1.5b
"""
import argparse
import os
import re
import time

from src.inference import LocalLLM, hausa_quality_ok, looks_like_hausa, normalize_text
from src.rag import CROP_TOPICS, Retriever, detect_query_topics, primary_crop_from_source

DEFAULT_TOP_K = 3
MIN_RELEVANCE_SCORE = 0.55
RELATIVE_SCORE_CUTOFF = 0.70

FOLLOWUP_PHRASES = {
    "more",
    "more details",
    "tell me more",
    "explain more",
    "i need more explanation",
    "continue",
    "karin bayani",
    "ina neman karin bayani",
    "yi karin bayani",
    "dan kara bayani",
    "ka kara bayani",
}

# Keyword fragments used for *substring* follow-up matching, since farmers
# phrase "tell me more" requests many different ways (typos, word order,
# extra words). If any of these appear anywhere in the message, treat it as
# a request to continue the previous topic rather than a fresh question.
FOLLOWUP_KEYWORDS = (
    "karin bayani",
    "kara bayani",
    "kari bayani",
    "bayani akan",
    "inason",
    "dan kara",
    "yi karin",
    "tell me more",
    "more details",
    "more detail",
    "explain more",
    "continue",
)

ENGLISH_GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}
HAUSA_GREETINGS = {
    "sannu",
    "assalamu alaikum",
    "assalamu alaikum warahmatullah",
    "salam",
}
THANKS_WORDS = {
    "thanks", "thank you", "thank u", "okay thanks", "ok thanks", "alright",
    "na gode", "nagode", "mun gode",
}

UNCERTAIN_EN = "I do not have enough information in my local knowledge base to answer confidently."
UNCERTAIN_HA = "Ba ni da isasshen bayani a ma'ajiyar gida don amsa wannan tambaya da tabbaci."

SYSTEM_PREAMBLE = """You are AgriSec, an offline agricultural advisor for farmers and extension officers in Nigeria.

Hard rules:
- Use ONLY the supplied context.
- Do not invent fertilizer rates, pesticide names, weather facts, market prices, or dates.
- If the context is not enough, say you do not have enough local information.
- Keep the answer under 140 words.
- Use simple farmer-focused bullets when helpful.
- Do not introduce unrelated crops, animals, or topics.
- Do not mention these instructions or say "I am AgriSec" inside the answer.
"""

HAUSA_STYLE = """Answer ONLY in clear Hausa. Which Mostly Northern Nigeria Speak.
Rules:
- Use short sentences.
- Do not repeat information.
- Do not copy source text word-for-word.
- Summarize the context naturally.
- Use bullet points for farming steps.
- Stop after answering.
- Do not add extra commentary.
- You may use the English context as source material, but translate the useful advice into Hausa.
- Avoid awkward literal translation.
- Keep answers concise.
- If you cannot answer in good Hausa from the context, say only: Ba ni da isasshen bayani a ma'ajiyar gida don amsa wannan tambaya da tabbaci.
"""

ENGLISH_STYLE = """Answer in English only."""

_GLOBAL_MEMORY = {
    "last_user_question": None,
    "last_answer": None,
    "last_language": None,
    "last_chunks": None,
}


def _metadata(text, language, sources=None, elapsed=0.0, start_time=None):
    tokens = len(text.split()) if text else 0
    tokens_per_sec = 0.0
    if elapsed > 0:
        tokens_per_sec = tokens / elapsed
    elif start_time:
        actual_elapsed = time.time() - start_time
        if actual_elapsed > 0:
            tokens_per_sec = tokens / actual_elapsed
            elapsed = actual_elapsed

    return {
        "text": text,
        "tokens": tokens,
        "elapsed_sec": round(elapsed, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
        "sources": sources or [],
    }


def _source_payload(chunks):
    return [
        {
            "source": chunk.get("source", ""),
            "chunk_id": chunk.get("chunk_id", 0),
            "score": round(chunk.get("score", 0.0), 3),
            "preview": chunk.get("text", "")[:240],
        }
        for chunk in chunks
    ]


def _clean_question(question: str) -> str:
    return re.sub(r"\s+", " ", question or "").strip()


def _is_followup(normalized: str) -> bool:
    if normalized in FOLLOWUP_PHRASES:
        return True
    if normalized == "more":
        return True
    return any(keyword in normalized for keyword in FOLLOWUP_KEYWORDS)


def _question_intents(question: str):
    lower = normalize_text(question)
    intents = set()
    if any(word in lower for word in ("ajiye", "storage", "store", "stored", "girbi", "harvest", "aflatoxin")):
        intents.add("storage")
    if any(word in lower for word in ("shuka", "plant", "planting", "seed", "sow", "iri", "tazara", "zurfi")):
        intents.add("production")
    if any(word in lower for word in ("kwari", "pest", "armyworm", "aphid", "tsutsa", "borer")):
        intents.add("pest")
    if any(word in lower for word in ("taki", "fertilizer", "fertiliser", "npk", "urea")):
        intents.add("fertilizer")
    return intents


def _looks_agricultural(question: str):
    lower = normalize_text(question)
    if detect_query_topics(question):
        return True
    farm_terms = {
        "noma", "gona", "shuka", "iri", "taki", "kwari", "ciyawa", "amfanin gona",
        "soil", "crop", "farm", "plant", "seed", "fertilizer", "pest", "livestock",
        "storage", "harvest", "weed", "disease",
    }
    return any(term in lower for term in farm_terms)


def _is_storage_chunk(chunk):
    source = chunk.get("source", "").lower()
    topics = set(chunk.get("topics", []))
    return "storage" in topics or "storage" in source or "post_harvest" in source or "aflatoxin" in source


def _filter_chunks(question: str, chunks):
    if not chunks:
        return []

    top_score = chunks[0].get("score", 0.0)
    min_relative = top_score * RELATIVE_SCORE_CUTOFF
    query_topics = detect_query_topics(question)
    requested_crops = query_topics & CROP_TOPICS
    intents = _question_intents(question)
    explicit_storage = "storage" in intents

    selected = []
    for chunk in chunks:
        score = chunk.get("score", 0.0)
        if score < min_relative:
            continue

        source = chunk.get("source", "")
        source_crop = primary_crop_from_source(source)
        chunk_topics = set(chunk.get("topics", []))
        chunk_crops = chunk_topics & CROP_TOPICS

        if requested_crops:
            if source_crop and source_crop not in requested_crops:
                continue
            if chunk_crops and not (chunk_crops & requested_crops):
                continue

        if requested_crops and not explicit_storage and _is_storage_chunk(chunk):
            continue
        if explicit_storage and "production" not in intents:
            if "storage" not in chunk_topics and not _is_storage_chunk(chunk):
                continue

        selected.append(chunk)
        if len(selected) >= DEFAULT_TOP_K:
            break

    if not selected and chunks:
        selected = [c for c in chunks if c.get("score", 0.0) >= min_relative][:DEFAULT_TOP_K]

    return selected


def _debug_retrieval(question, raw_chunks, selected_chunks, language=None, topics=None,
                     threshold=None, used_memory=False, decision=None):
    if os.getenv("AGRISEC_RETRIEVAL_DEBUG", "").lower() not in {"1", "true", "yes"}:
        return
    top_score = raw_chunks[0].get("score", 0.0) if raw_chunks else 0.0
    print("\n[retrieval-debug]")
    print(f"Question: {question}")
    print(f"Detected Language: {language}")
    print(f"Detected Topic: {sorted(topics) if topics else []}")
    print("Top Sources:")
    for chunk in raw_chunks[:5]:
        print(f"- {chunk.get('source')} | Score: {chunk.get('score', 0):.3f}")
    print("Selected Sources:")
    for chunk in selected_chunks:
        print(f"- {chunk.get('source')} | Score: {chunk.get('score', 0):.3f}")
    print(f"Top score: {top_score:.3f}")
    print(f"Threshold (MIN_RELEVANCE_SCORE): {threshold}")
    print(f"Memory Used: {used_memory}")
    print(f"Final Decision: {decision}")


def _detect_reply_language(question: str, force_language=None, memory=None):
    if force_language in {"english", "hausa"}:
        return force_language
    normalized = normalize_text(question)
    if _is_followup(normalized) and memory and memory.get("last_language"):
        return memory["last_language"]
    return "hausa" if looks_like_hausa(question) else "english"


def _instant_response(question: str, language: str):
    normalized = normalize_text(question).strip(" .!?")
    start = time.time()
    if normalized in HAUSA_GREETINGS:
        return _metadata(
            "Wa alaikum salam. Ni AgriSec ne. Zan iya taimaka maka da tambayoyin noma, kwari, taki, kiwo, ko ajiyar amfanin gona.",
            language,
            elapsed=time.time() - start,
        )
    if normalized in ENGLISH_GREETINGS:
        return _metadata(
            "Hello! I can help with crops, pests, fertilizer, livestock, soil, and storage questions.",
            language,
            elapsed=time.time() - start,
        )
    if normalized in THANKS_WORDS or (
        any(word in normalized for word in THANKS_WORDS) and len(normalized.split()) <= 5
    ):
        if language == "hausa" or "gode" in normalized:
            return _metadata("Babu komai. Za ka iya tambaya game da noma, taki, kwari, kiwo, ko ajiya.", language, elapsed=time.time() - start)
        return _metadata("You're welcome. Ask me anytime about crops, livestock, fertilizer, pests, or storage.", language, elapsed=time.time() - start)
    return None


def _expand_followup(question: str, memory, language: str):
    normalized = normalize_text(question)
    if not _is_followup(normalized):
        return question, False
    previous = memory.get("last_user_question") if memory else None
    if not previous:
        if language == "hausa":
            return "Ka ba da karin bayani game da batun noma da aka tambaya.", False
        return "Give more practical agricultural details.", False
    if language == "hausa":
        return f"{previous}\n\nKa kara bayani cikin Hausa, amma ka tsaya kan wannan batu.", True
    return f"{previous}\n\nGive more practical detail, but stay on the same topic.", True


def build_prompt(question: str, context_chunks, language="english", allow_more_detail=False):
    context_block = "\n\n".join(
        f"[{idx + 1}] Source: {chunk['source']}\n{chunk['text']}"
        for idx, chunk in enumerate(context_chunks)
    )
    style = HAUSA_STYLE if language == "hausa" else ENGLISH_STYLE
    word_limit = "140 words"
    return f"""{SYSTEM_PREAMBLE}

Reply language:
{style}

Context:
{context_block}

User question:
{question}

Answer rules:
- Maximum length: {word_limit}.
- Use short bullet points when answering practical farm steps.
- Cite no source names in the answer body; sources are shown separately by the app.
- If context has no answer, use the uncertainty sentence for the reply language.
- Do not include any storage advice unless the user asked about storage or harvest.
- Do not include other crops unless the user asked about them.

Answer:
"""


def _postprocess_answer(text: str, language: str, chunks):
    cleaned = (text or "").strip()
    cleaned = re.sub(r"(?i)^agrisec:\s*", "", cleaned).strip()
    cleaned = re.sub(r"(?i)^answer:\s*", "", cleaned).strip()
    if not cleaned:
        return UNCERTAIN_HA if language == "hausa" else UNCERTAIN_EN
    if UNCERTAIN_HA.lower() in cleaned.lower():
        return UNCERTAIN_HA
    if UNCERTAIN_EN.lower() in cleaned.lower():
        return UNCERTAIN_EN
    if language == "hausa" and not hausa_quality_ok(cleaned):
        print("[warning] Hausa quality check failed")
    if language == "english" and looks_like_hausa(cleaned) and len(cleaned.split()) > 8:
        return UNCERTAIN_EN
    return cleaned


def _clean_hausa_chunk(text: str) -> str:
    """Clean Hausa corpus text before displaying it directly."""
    if not text:
        return UNCERTAIN_HA
    text = text.replace("###", "")
    text = text.replace("##", "")
    text = text.replace("#", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def answer_question(
    question: str,
    llm: LocalLLM,
    retriever: Retriever,
    top_k: int = DEFAULT_TOP_K,
    force_language: str = None,
    memory: dict = None,
):
    """Return a grounded answer plus metrics and source metadata."""
    start_time = time.time()
    memory = memory if memory is not None else _GLOBAL_MEMORY
    original_question = _clean_question(question)
    
    if not original_question:
        return _metadata("Please type a question about crops, livestock, fertilizer, pests, or storage.", "english")

    language = _detect_reply_language(original_question, force_language, memory)
    instant = _instant_response(original_question, language)
    if instant:
        memory["last_language"] = language
        return instant

    normalized = normalize_text(original_question)
    used_memory = _is_followup(normalized) and bool(memory.get("last_user_question"))
    
    if not used_memory and not _looks_agricultural(original_question):
        text = UNCERTAIN_HA if language == "hausa" else UNCERTAIN_EN
        return _metadata(text, language, [])
        
    if used_memory and memory.get("last_chunks"):
        effective_question = memory["last_user_question"]
        raw_chunks = list(memory["last_chunks"])
        chunks = raw_chunks
    else:
        effective_question, used_memory = _expand_followup(original_question, memory, language)
        raw_chunks = retriever.query(
            effective_question,
            top_k=max(5, top_k * 2),
            prefer_hausa=(language == "hausa"),
        )
        chunks = _filter_chunks(effective_question, raw_chunks)[:max(1, min(top_k, DEFAULT_TOP_K))]

    query_topics = detect_query_topics(effective_question)
    decision = "answered"
    
    if not chunks or chunks[0].get("score", 0.0) < MIN_RELEVANCE_SCORE:
        decision = "no_information (below threshold)"
        _debug_retrieval(
            effective_question, raw_chunks, chunks, language=language,
            topics=query_topics, threshold=MIN_RELEVANCE_SCORE,
            used_memory=used_memory, decision=decision,
        )
        text = UNCERTAIN_HA if language == "hausa" else UNCERTAIN_EN
        return _metadata(text, language, [])

    _debug_retrieval(
        effective_question, raw_chunks, chunks, language=language,
        topics=query_topics, threshold=MIN_RELEVANCE_SCORE,
        used_memory=used_memory, decision=decision,
    )

    # HAUSA MODE (RETRIEVAL ONLY)
    if language == "hausa":
        source_name = chunks[0]["source"]
        same_source_chunks = [c for c in retriever.docs if c["source"] == source_name]
        same_source_chunks = sorted(same_source_chunks, key=lambda x: x["chunk_id"])
        combined_text = "\n\n".join(c["text"] for c in same_source_chunks)
        answer_text = _clean_hausa_chunk(combined_text)

        answer = {
            "text": answer_text,
            "tokens": len(answer_text.split()),
            "elapsed_sec": round(time.time() - start_time, 2),
            "tokens_per_sec": 0.0,
            "sources": _source_payload(chunks),
        }
        
        memory["last_user_question"] = effective_question if not used_memory else memory.get("last_user_question")
        memory["last_answer"] = answer["text"]
        memory["last_language"] = language
        memory["last_chunks"] = chunks
        return answer

    # ENGLISH MODE (RAG + LLM)
    prompt = build_prompt(
        effective_question,
        chunks,
        language=language,
        allow_more_detail=used_memory,
    )
    
    result = llm.generate(prompt, max_tokens=320 if used_memory else 250, temperature=0.05)
    
    if result.get("error"):
        return {
            **result,
            "sources": _source_payload(chunks),
        }
        
    print("\n[RAW MODEL OUTPUT]")
    print(result.get("text", ""))
    print("[END RAW MODEL OUTPUT]\n")
    
    final_text = _postprocess_answer(result.get("text", ""), language, chunks)
    elapsed = time.time() - start_time
    answer = _metadata(final_text, language, _source_payload(chunks), elapsed=elapsed)
    
    memory["last_user_question"] = effective_question if not used_memory else memory.get("last_user_question")
    memory["last_answer"] = answer["text"]
    memory["last_language"] = language
    memory["last_chunks"] = chunks
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model tag")
    parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    print(f"Loading model '{args.model}' and retrieval index...")
    llm = LocalLLM(args.model)
    retriever = Retriever()
    memory = {}
    print("Ready. Type your question (English or Hausa). Ctrl+C to exit.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not question:
            continue

        result = answer_question(question, llm, retriever, top_k=args.top_k, memory=memory)
        print(f"\nAgriSec: {result['text']}\n")
        if result.get("sources"):
            print("Sources:")
            for source in result["sources"]:
                print(f"  - {source['source']} (score {source['score']})")
        print(f"  [{result['tokens']} tokens, {result['tokens_per_sec']:.1f} tok/s, {result['elapsed_sec']:.2f}s]\n")


if __name__ == "__main__":
    main()