# AgriSec - Working Plan

## Current Status

The project is now a stabilized ADTC Agriculture submission scaffold with:

- Ollama/Qwen2.5:1.5B local inference
- FAISS RAG over `data/corpus`
- multilingual sentence-transformers retrieval
- English/Hausa language routing
- greeting and thanks shortcuts
- lightweight follow-up memory
- localhost web UI for judging/demo UX
- benchmark script for TPS, latency, peak RAM, Sperf, and Seff estimates

## Decisions Already Made

- **Domain:** Agriculture.
- **Model:** `qwen2.5:1.5b` through Ollama.
- **Retrieval:** FAISS + `paraphrase-multilingual-MiniLM-L12-v2`.
- **Interface:** Use the local web UI as the primary demo, keep CLI for testing.
- **Dependency posture:** Avoid heavy UI frameworks to preserve RAM.

## Remaining Decisions

1. **Benchmark confirmation:** Run on target 8 GB-class laptop and record real numbers.
2. **Hausa review:** Have Hausa corpus files reviewed by a native speaker or extension officer.
3. **Thermal behavior:** Test with official ADTC profiler or system sensors.
4. **Submission identity:** Team name, member names, GitHub URL, and demo video link.

## Implemented

- [x] Local RAG corpus
- [x] FAISS index builder/query tool
- [x] Ollama inference wrapper
- [x] Prompt safety layer
- [x] Hausa detection and Hausa answer enforcement
- [x] Greeting/thanks shortcuts
- [x] Follow-up memory for "more details" / "karin bayani"
- [x] Local web UI with quick topics, sources, language mode, and metrics
- [x] Benchmark script
- [x] README, architecture notes, and report draft

## Run Locally

```bash
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
python -m src.rag --build
python -m src.web_app --model qwen2.5:1.5b
python scripts/benchmark.py --model qwen2.5:1.5b --runs 5
```

Record:

- Mean tokens/sec
- First response time
- Python process peak RAM
- Ollama model RAM from `ollama ps`
- Max temperature or throttle status
- Two English Q&A pairs with sources
- Two Hausa Q&A pairs with sources

## Submission Polish

- [ ] Fill all `[TBD]` fields in `reports/REPORT.md`
- [ ] Add screenshots to `assets/`
- [ ] Record max 2-minute demo video
- [ ] Regenerate `reports/REPORT.docx`
- [ ] Push to GitHub
- [ ] Check against the official ADTC template and profiler output

## Suggested Next Corpus Additions

- Hausa fall armyworm and aphids guide
- Hausa fertilizer safety guide
- Hausa livestock basics
- Cassava and tomato Hausa summaries
- Short seasonal/weather advisory for Northern Nigeria
