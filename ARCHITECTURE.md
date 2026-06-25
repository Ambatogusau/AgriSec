# AgriSec Local Assistant - System Architecture

## Architectural Goals

- Run offline on an ADTC-style 8 GB RAM laptop.
- Keep latency low with instant paths for greetings/thanks and short model responses.
- Keep answers grounded in local agriculture files.
- Support English, Hausa, and mixed Hausa/English queries.
- Make UI/UX demo-ready with sources, metrics, and user-friendly errors.

## Implemented Architecture

```text
Browser at localhost:7860 or CLI
        |
        v
src/assistant.py
language detection, greetings, memory, prompt safety, answer cleanup
        |
        +-----------------------------+
        |                             |
        v                             v
src/rag.py                       src/inference.py
FAISS + multilingual             Ollama API wrapper
MiniLM embeddings                qwen2.5:1.5b
        |                             |
        v                             v
data/corpus/                     local Ollama model store
```

## Component Details

### Presentation Layer

`src/web_app.py` serves a local web UI with no CDN, build step, or external backend. It includes quick-topic buttons, language mode, offline/model status, source citations, response metrics, and session memory through the running `RealEngine`.

### Application Layer

`src/assistant.py` handles:

- deterministic greeting/thanks responses before retrieval
- Hausa/English/mixed language routing
- lightweight memory for "more details" / "karin bayani" follow-ups
- strict prompt rules for grounded answers under 100 words
- Hausa quality checks and fallback to Hausa corpus text when the model responds poorly
- uncertainty answers when retrieval relevance is weak

### Retrieval Layer

`src/rag.py` cleans corpus documents, strips internal team notes and source footers from retrieval chunks, removes duplicate chunks, embeds with `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, and reranks FAISS results using:

- semantic score
- keyword overlap
- topic overlap
- Hausa-source boost for Hausa queries
- penalty for unrelated topics

### Inference Layer

`src/inference.py` calls Ollama's local `/api/generate` endpoint with `qwen2.5:1.5b`. Defaults are tuned for the 8 GB target: `num_ctx=1536`, `num_predict` near 90 by default, low temperature, repeat penalty, and four CPU threads.

### Benchmark Layer

`scripts/benchmark.py` warms up Ollama, measures tokens/sec, first response time, Python process RSS, and estimates Sperf/Seff. Its RAM note is explicit: Ollama model memory must also be checked with `ollama ps` or the official ADTC profiler.

## Scoring Alignment

| Scoring area | Architecture support |
|---|---|
| Accuracy | RAG grounding, source filtering, uncertainty gate, reduced hallucination prompt |
| Speed | greeting shortcut, lower `top_k`, short context, lower token cap |
| RAM efficiency | Qwen2.5:1.5B via Ollama, no heavy frontend framework |
| Thermal | conservative token/thread settings; still needs hardware test |
| UI/UX | local web UI, quick topics, bilingual mode, sources, metrics, friendly errors |

## Known Limitations

- The UI does not stream tokens yet.
- Hausa output quality depends on both Qwen and available Hausa source files.
- Voice input/output is not implemented.
- Final thermal and system-wide memory measurements must come from target hardware.
