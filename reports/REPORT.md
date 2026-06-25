# AgriSec Local Assistant
### Africa Deep Tech Challenge 2026 - Agriculture Domain

**Team name:** `[TBD]`  
**Team members:** `[TBD]`  
**Repository:** `[TBD - GitHub URL]`  
**Demo video:** `[TBD - link, max 2 minutes]`

---

## 1. Problem Definition and Context

Smallholder farmers and agricultural extension officers often need fast, practical advice on crop production, pest control, fertilizer use, livestock care, and post-harvest storage. Cloud AI tools can be unreliable in low-connectivity settings because they require internet access, API fees, and stable power.

AgriSec Local Assistant is an offline agricultural advisory tool for the ADTC Agriculture domain. It runs locally through Ollama, retrieves from a local agriculture corpus, and answers with visible source files so advice can be inspected.

## 2. Identified Constraints

| Constraint | Description |
|---|---|
| Compute | CPU-oriented inference on the ADTC Standard Laptop profile |
| Memory | Must remain suitable for 8 GB RAM laptops and the 7 GB scoring budget |
| Connectivity | No cloud APIs or internet during inference after setup |
| Power | Avoid long responses and sustained load; final thermal test still TBD |
| Language | English and Hausa support, including mixed queries |
| Data | Corpus files must be redistributable and traceable |
| UI/UX | Judges need a clear interface with fast feedback and source visibility |

## 3. Design Alternatives and Final Decisions

| Decision point | Alternatives considered | Current choice | Reasoning |
|---|---|---|---|
| Base model | Llama 3.2 1B/3B, Gemma 2B, Qwen 1.5B/3B | Qwen2.5:1.5B via Ollama | Small, practical, easy local deployment |
| Retrieval | FAISS, Chroma, no RAG | FAISS CPU | Lightweight, local, no server process |
| Embeddings | English MiniLM, multilingual MiniLM | `paraphrase-multilingual-MiniLM-L12-v2` | Better English/Hausa retrieval |
| Interface | CLI only, desktop shell, browser UI | Lightweight localhost web UI plus CLI | Good judging UX without heavy RAM cost |
| Backend | FastAPI, Flask, Python stdlib server | Python stdlib `http.server` | Fewer dependencies and easier offline setup |

## 4. Tools Used and Why

- **Ollama** - local model runtime with simple offline deployment after pulling the model.
- **Qwen2.5:1.5B** - small enough for commodity laptops while still useful for RAG.
- **FAISS CPU** - local vector search over agriculture documents.
- **sentence-transformers** - multilingual embeddings for English and Hausa retrieval.
- **Python standard-library HTTP server** - lightweight UI server with no extra web dependency.
- **psutil** - benchmark support for process memory and timing.

## 5. Implementation Summary

- `src/inference.py` - Ollama wrapper, language utilities, Hausa quality guard.
- `src/rag.py` - corpus cleaning, chunk dedupe, FAISS build/query, relevance reranking.
- `src/assistant.py` - greetings, thanks, memory, prompt, uncertainty gate, answer cleanup.
- `src/web_app.py` - offline localhost UI with quick topics, language mode, sources, and metrics.
- `scripts/benchmark.py` - Ollama benchmark script for TPS, latency, peak RAM, Sperf, and Seff estimates.

Run:

```bash
ollama pull qwen2.5:1.5b
python -m src.rag --build
python -m src.web_app --model qwen2.5:1.5b
```

## 6. Performance Tests and Benchmarks

> Replace this table with real output from `scripts/benchmark.py` and `ollama ps` on the target laptop.

| Metric | Value | Notes |
|---|---|---|
| Model | Qwen2.5:1.5B | Ollama tag: `qwen2.5:1.5b` |
| Mean TPS | `[TBD]` tok/s | Sperf = 100 * (TPS / 15.0) |
| First response time | `[TBD]` s | Includes Ollama warmup/load |
| Python peak RAM | `[TBD]` GB | From benchmark script |
| Ollama model RAM | `[TBD]` GB | From `ollama ps` or ADTC profiler |
| Thermal / throttle observed | `[TBD]` | Use ADTC profiler or `sensors` |
| Qualitative accuracy notes | `[TBD]` | Include English and Hausa examples |

Benchmark command:

```bash
python scripts/benchmark.py --model qwen2.5:1.5b --runs 5
```

## 7. Example Interactions

```text
Q: How do I control fall armyworm in my maize farm without expensive pesticides?
A: [TBD - paste real model answer]
Sources: [TBD]
```

```text
Q: Explain in Hausa how to store maize safely after harvest.
A: [TBD - paste real model answer]
Sources: [TBD]
```

## 8. African Use Case Relevance

AgriSec focuses on agriculture advice for regions where farmers and extension officers may not have reliable internet access. The strongest use-case angle is offline laptop deployment through extension offices, schools, hubs, and cooperatives, with Hausa-oriented support for Northern Nigeria and source-grounded advice that officers can inspect.

## 9. UI/UX Summary

Current UI features:

- Quick-topic buttons for common farmer workflows
- Large text input and keyboard-friendly send action
- Reply language selector: auto, English, Hausa
- Offline/model status indicator
- Source list after each model answer
- Per-answer metrics: tokens, tokens/sec, elapsed time
- Friendly errors if Ollama or the RAG index is missing

## 10. Screenshots / Video

- Screenshots: `[TBD - add to assets/]`
- Demo video: `[TBD - link, max 2 minutes]`

Suggested demo flow:

1. Show Ollama running locally and the app on `127.0.0.1:7860`.
2. Ask one English agriculture question.
3. Show source citations and tokens/sec.
4. Ask one Hausa-oriented question.
5. Ask "karin bayani" to show memory.
6. End with benchmark numbers and why it fits the ADTC laptop constraint.

## 11. Remaining Submission Tasks

- Fill every `[TBD]` field with real data.
- Add screenshots and final demo video.
- Regenerate `REPORT.docx` from this Markdown after final edits.
- Run the official ADTC profiler and compare with `scripts/benchmark.py`.
- Review Hausa corpus content with a native speaker or extension officer.
