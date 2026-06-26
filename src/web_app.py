import argparse
import json
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AgriSec Local Assistant</title>
  <style>
    :root {
      --bg: #f4f7f2;
      --panel: #ffffff;
      --ink: #172116;
      --muted: #5f6f5b;
      --line: #d7e2d2;
      --green: #1d6b3a;
      --green-dark: #104526;
      --amber: #b56a16;
      --danger: #8b2e22;
      font-family: Arial, Helvetica, sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
    }
    .shell {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: #eef5eb;
      padding: 20px;
    }
    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.85);
      padding: 16px 22px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      line-height: 1.1;
      letter-spacing: 0;
    }
    .subtitle {
      color: var(--muted);
      font-size: 13px;
      margin-top: 6px;
    }
    .status {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      background: var(--panel);
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .pill.good { color: var(--green-dark); border-color: #a9c9aa; }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      margin-bottom: 24px;
    }
    .mark {
      display: grid;
      place-items: center;
      width: 46px;
      height: 46px;
      border-radius: 8px;
      background: var(--green);
      color: white;
      font-weight: 700;
      font-size: 18px;
    }
    .brand strong { display: block; font-size: 19px; }
    .brand span { color: var(--muted); font-size: 12px; }
    .side-title {
      margin: 20px 0 10px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .quick-grid {
      display: grid;
      gap: 10px;
    }
    button, select, textarea {
      font: inherit;
    }
    .quick {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 12px;
      text-align: left;
      cursor: pointer;
      min-height: 48px;
    }
    .quick:hover, .quick:focus { border-color: var(--green); outline: none; }
    .control {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      padding: 11px 12px;
    }
    .chat {
      overflow-y: auto;
      padding: 24px;
    }
    .empty {
      max-width: 760px;
      margin: 10vh auto 0;
    }
    .empty h2 {
      font-size: 32px;
      line-height: 1.15;
      margin: 0 0 12px;
      letter-spacing: 0;
    }
    .empty p {
      color: var(--muted);
      line-height: 1.55;
      margin: 0 0 18px;
    }
    .examples {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
      margin-top: 20px;
    }
    .message {
      max-width: 820px;
      margin: 0 auto 16px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 15px 16px;
      line-height: 1.55;
      white-space: pre-wrap;
    }
    .message.user {
      border-color: #b9d3ba;
      background: #f8fbf7;
    }
    .message .label {
      display: block;
      margin-bottom: 6px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .sources {
      max-width: 820px;
      margin: -8px auto 16px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .sources strong { color: var(--ink); }
    form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 16px 22px;
    }
    textarea {
      resize: none;
      min-height: 52px;
      max-height: 140px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      color: var(--ink);
    }
    textarea:focus { border-color: var(--green); outline: none; }
    .send {
      border: 0;
      border-radius: 8px;
      min-width: 110px;
      background: var(--green);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    .send:disabled {
      background: #8ba68d;
      cursor: wait;
    }
    .metrics {
      color: var(--amber);
      font-size: 12px;
      margin-top: 8px;
    }
    .lang-badge{
    display:inline-block;
    margin-top:8px;
    padding:6px 12px;
    border-radius:20px;
    background:#e8f5e9;
    color:#1b5e20;
    font-size:.8rem;
    font-weight:600;
}
.lang-note{
    margin-top:10px;
    padding:10px 12px;
    background:#f3faf4;
    border-left:4px solid #2e7d32;
    border-radius:8px;
    font-size:.82rem;
    line-height:1.5;
    color:#35523c;
}
    .error { color: var(--danger); }
    @media (max-width: 800px) {
      .shell { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { min-height: auto; }
      header { align-items: flex-start; flex-direction: column; }
      .status { justify-content: flex-start; }
      .examples { grid-template-columns: 1fr; }
      form { grid-template-columns: 1fr; }
      .send { min-height: 48px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <div class="mark">AS</div>
        <div>
          <strong>AgriSec</strong>
          <span>Offline farm advisory</span>
        </div>
      </div>
      <label class="side-title" for="language">Reply language</label>
<select id="language" class="control">
  <option value="auto">Auto detect</option>
  <option value="english">English</option>
  <option value="hausa">Hausa</option>
</select>

<div class="lang-note">
  💡 <strong>Hausa:</strong> Kuna iya yin tambayoyi da Hausa ko Turanci. AgriSec zai gane harshen ku ta atomatik.
</div>
      <div class="side-title">Quick topics</div>
      <div class="quick-grid">
        <button class="quick" data-q="How do I control fall armyworm in my maize farm without expensive pesticides?">Pest control</button>
        <button class="quick" data-q="Recommend a simple fertilizer schedule for rice on loamy soil.">Fertilizer plan</button>
        <button class="quick" data-q="What is the best planting window for maize in Northern Nigeria?">Planting window</button>
        <button class="quick" data-q="How can I prevent aflatoxin after harvest?">Storage safety</button>
      </div>
      <div class="side-title">Judge checklist</div>
      <div class="pill good">Localhost only</div>
      <div class="pill good">Source citations</div>
      <div class="pill good">CPU inference</div>
    </aside>
    <main>
      <header>
        <div class="subtitle">
  Agriculture domain assistant for ADTC 2026. No cloud calls at inference time.
</div>

<div class="lang-badge">
  🌍 English • Hausa • 100% Offline
</div>
        <div class="status">
          <span class="pill good" id="mode">Offline ready</span>
          <span class="pill" id="model">Model: loading</span>
        </div>
      </header>
      <section class="chat" id="chat">
        <div class="empty" id="empty">
          <h2>Ask practical crop, pest, fertilizer, livestock, or storage questions.</h2>
          <p>Answers are grounded in the local corpus and show the files used, so extension officers can inspect where advice came from during a demo or field session.</p>
          <div class="examples">
            <button class="quick" data-q="My cowpea leaves have small insects under them. What should I do first?">Cowpea pest triage</button>
            <button class="quick" data-q="Explain in Hausa how to store maize safely after harvest.">Hausa storage advice</button>
          </div>
        </div>
      </section>
      <form id="form">
      <textarea id="question" placeholder="Type a farmer question in English/Hausa..." aria-label="Question"></textarea>
        <button class="send" id="send" type="submit">Send</button>
      </form>
    </main>
  </div>
  <script>
    const chat = document.getElementById("chat");
    const empty = document.getElementById("empty");
    const form = document.getElementById("form");
    const question = document.getElementById("question");
    const send = document.getElementById("send");
    const language = document.getElementById("language");
    const model = document.getElementById("model");
    const mode = document.getElementById("mode");

    fetch("/api/status").then(r => r.json()).then(data => {
      model.textContent = "Model: " + data.model;
      mode.textContent = data.demo ? "Demo UI mode" : "Offline ready";
    });

    function addMessage(role, text, metrics) {
      empty.style.display = "none";
      const div = document.createElement("div");
      div.className = "message " + role;
      const label = document.createElement("span");
      label.className = "label";
      label.textContent = role === "user" ? "You" : "AgriSec";
      div.appendChild(label);
      div.appendChild(document.createTextNode(text));
      if (metrics) {
        const m = document.createElement("div");
        m.className = "metrics";
        m.textContent = metrics;
        div.appendChild(m);
      }
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
      return div;
    }

    function addSources(sources) {
      if (!sources || !sources.length) return;
      const div = document.createElement("div");
      div.className = "sources";
      div.innerHTML = "<strong>Sources:</strong> " + sources.map(s =>
        `${s.source} (${s.score})`
      ).join(", ");
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    async function ask(text) {
      const value = text.trim();
      if (!value) return;
      addMessage("user", value);
      question.value = "";
      send.disabled = true;
      send.textContent = "Working";
      const pending = addMessage("assistant", "Thinking with local corpus...");
      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({question: value, language: language.value})
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || "Request failed");
        pending.childNodes[pending.childNodes.length - 1].nodeValue = data.text || "No answer returned.";
        addSources(data.sources);
        const metrics = `${data.tokens || 0} tokens, ${(data.tokens_per_sec || 0).toFixed(1)} tok/s, ${(data.elapsed_sec || 0).toFixed(2)}s`;
        const m = document.createElement("div");
        m.className = "metrics";
        m.textContent = metrics;
        pending.appendChild(m);
      } catch (err) {
        pending.classList.add("error");
        pending.childNodes[pending.childNodes.length - 1].nodeValue = err.message;
      } finally {
        send.disabled = false;
        send.textContent = "Send";
      }
    }

    document.querySelectorAll("[data-q]").forEach(btn => {
      btn.addEventListener("click", () => ask(btn.dataset.q));
    });
    form.addEventListener("submit", event => {
      event.preventDefault();
      ask(question.value);
    });
    question.addEventListener("keydown", event => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });
  </script>
</body>
</html>"""


class DemoEngine:
    def __init__(self):
        self.model_label = "demo responses"
        self.memory = {}

    def answer(self, question, language):
        start = time.time()
        lower = question.lower()
        if "hausa" in lower or language == "hausa":
            text = (
                "Ajiye amfanin gona a wuri mai bushe da iska. Ka busar da masara sosai "
                "kafin a saka ta a buhu, ka cire hatsi masu lalacewa, sannan ka duba "
                "ajiyar duk mako domin hana kwari da aflatoxin."
            )
        elif "fall armyworm" in lower or "maize" in lower:
            text = (
                "Scout the maize field early in the morning, remove badly damaged whorls "
                "where practical, and use locally approved biopesticide or selective "
                "insecticide only when damage passes the local action threshold. Keep the "
                "field weed-free and avoid spraying broad-spectrum chemicals too often."
            )
        else:
            text = (
                "Start with the local crop guide, confirm the crop stage, soil condition, "
                "and symptoms, then choose the least-cost action first. If the local corpus "
                "does not cover the case, AgriSec should say so instead of guessing."
            )
        elapsed = time.time() - start
        tokens = max(12, len(text.split()))
        return {
            "text": text,
            "tokens": tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec": tokens / elapsed if elapsed else 0.0,
            "sources": [
                {
                    "source": "demo_mode.md",
                    "chunk_id": 0,
                    "score": 1.0,
                    "preview": "Demo mode uses sample responses when Ollama is not loaded.",
                }
            ],
        }


class RealEngine:
    def __init__(self, model_name, top_k):
        from src.inference import LocalLLM
        from src.rag import Retriever

        self.model_label = model_name
        self.llm = LocalLLM(model_name)
        self.retriever = Retriever()
        self.top_k = top_k
        self.memory = {}

    def answer(self, question, language):
        from src.assistant import answer_question

        force_language = language if language in {"english", "hausa"} else None
        return answer_question(
            question,
            self.llm,
            self.retriever,
            top_k=self.top_k,
            force_language=force_language,
            memory=self.memory,
        )


def make_handler(engine, demo):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

        def _json(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                data = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif path == "/api/status":
                self._json(200, {"model": engine.model_label, "demo": demo})
            else:
                self._json(404, {"error": "Not found"})

        def do_POST(self):
            path = urlparse(self.path).path
            if path != "/api/chat":
                self._json(404, {"error": "Not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or "{}")
                question = (payload.get("question") or "").strip()
                language = payload.get("language") or "auto"
                if not question:
                    self._json(400, {"error": "Question is required"})
                    return
                self._json(200, engine.answer(question, language))
            except Exception as exc:
                self._json(
                    500,
                    {
                        "error": (
                            "AgriSec could not answer this request. If you just rebuilt "
                            "the corpus, restart the app and make sure Ollama is running. "
                            f"Details: {exc}"
                        )
                    },
                )

    return Handler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5:1.5b", help="Ollama model tag")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--demo", action="store_true", help="Run UI with sample responses")
    parser.add_argument("--open", action="store_true", help="Open the UI in the default browser")
    args = parser.parse_args()

    if args.demo:
        engine = DemoEngine()
    else:
        engine = RealEngine(args.model, args.top_k)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine, args.demo))
    url = f"http://{args.host}:{args.port}"
    print(f"AgriSec web UI running at {url}")
    print("Press Ctrl+C to stop.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
        server.server_close()


if __name__ == "__main__":
    main()
