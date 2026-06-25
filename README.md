# AgriSec Local Assistant

Offline, on-device agricultural advisory assistant for the **Africa Deep Tech Challenge 2026** Agriculture domain. AgriSec uses **Ollama + Qwen2.5:1.5B**, FAISS, and sentence-transformers to run a local RAG system on commodity 8 GB laptops.

## What This Project Builds

AgriSec combines:

- Ollama local inference with `qwen2.5:1.5b`
- Retrieval-augmented generation over a local agriculture corpus
- English, Hausa, and mixed-query language handling
- Lightweight conversation memory for follow-up questions
- A localhost web UI for demos and field-style usage
- Benchmark tooling for tokens/sec, latency, and peak RAM

## Architecture

```text
User question (English or Hausa)
        |
        v
Local web UI or CLI
        |
        v
Language/greeting/memory/safety layer
        |
        v
Retriever (multilingual sentence-transformers + FAISS)
        |
        v
Ollama model: qwen2.5:1.5b
        |
        v
Grounded answer + sources + speed metrics
```

## Why These Choices Fit ADTC

| Hackathon requirement | AgriSec alignment |
|---|---|
| Runs on commodity laptops | Small Qwen2.5:1.5B model through Ollama, short context, low token limit |
| No cloud dependency | Local Ollama model, local FAISS index, local corpus, localhost UI |
| Agriculture domain | Corpus covers crops, pests, fertilizer, livestock, storage, and Hausa guides |
| Accuracy | RAG grounding, relevance gate, source filtering, concise safety prompt |
| Speed | Instant greeting/thanks path, top-k capped at 3, short responses by default |
| RAM efficiency | No heavy frontend framework; one small embedding model and one small LLM |
| UI/UX judged | Web UI includes quick topics, bilingual mode, offline status, sources, and metrics |

## Repo Structure

```text
agrisec/
├── src/
│   ├── assistant.py      # Language, memory, prompt, safety, orchestration
│   ├── inference.py      # Ollama/Qwen wrapper and language utilities
│   ├── rag.py            # Corpus cleaning, chunking, FAISS retrieval/reranking
│   └── web_app.py        # Lightweight offline localhost UI
├── scripts/
│   └── benchmark.py      # Ollama TPS / peak RAM / latency profiler
├── data/
│   ├── corpus/           # Local agriculture documents
│   └── index/            # Generated FAISS index, not committed
├── reports/
│   ├── REPORT.md         # ADTC project report draft
│   └── REPORT.docx       # Regenerate after final Markdown edits
├── assets/               # Add screenshots and demo video here
└── requirements.txt
```

## Setup on the Target Laptop

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

ollama pull qwen2.5:1.5b
python -m src.rag --build
```

Run the web UI:

```bash
python -m src.web_app --model qwen2.5:1.5b
```

Open [http://127.0.0.1:7860](http://127.0.0.1:7860). The app binds to localhost only.

Preview the UI without Ollama:

```bash
python -m src.web_app --demo
```

Run the CLI assistant:

```bash
python -m src.assistant --model qwen2.5:1.5b
```

Run benchmarks:

```bash
python scripts/benchmark.py --model qwen2.5:1.5b --runs 5
```

## Submission Checklist

- Run `python -m src.rag --build` after every corpus or retrieval change.
- Run `scripts/benchmark.py` on target hardware and paste real numbers into `reports/REPORT.md`.
- Capture UI screenshots in `assets/`.
- Record a max 2-minute demo video showing offline mode, English and Hausa answers, sources, and benchmark numbers.
- Regenerate `reports/REPORT.docx` from the final report.

## Current Gaps Before Submission

- Final benchmark numbers are still required from an ADTC-style laptop.
- Hausa documents should be reviewed by a native Hausa speaker before real farmer deployment.
- Add more Hausa corpus files for pests, fertilizer, and livestock to improve Hausa coverage.
