# 🌾 AgriSec

## Offline AI Agricultural Assistant for African Farmers

AgriSec is an AI-powered offline agricultural assistant developed for the **Africa Deep Tech Challenge (ADTC) 2026**. It helps farmers access practical agricultural knowledge without requiring an internet connection.

The system combines **Retrieval-Augmented Generation (RAG)**, **Ollama**, **Qwen2.5**, **FAISS**, and a multilingual agricultural knowledge base to answer farming questions in **English** and **Hausa** while running completely on a standard laptop.

---

# 🚜 The Problem

Millions of farmers across Africa have limited access to agricultural extension services.

Many rural communities experience:

* Poor or no internet connectivity
* Limited access to agricultural experts
* Language barriers
* Difficulty accessing reliable farming information

Most AI assistants depend on cloud services, making them difficult to use in remote farming communities.

---

# 💡 Our Solution

AgriSec brings agricultural intelligence completely offline.

Instead of relying on cloud APIs, AgriSec stores agricultural knowledge locally and uses AI to retrieve the most relevant information before generating accurate, source-backed responses.

The system can assist farmers with:

* 🌽 Crop production
* 🌱 Fertilizer recommendations
* 🐛 Pest and disease management
* 🐄 Livestock management
* 🌾 Post-harvest storage
* 🌍 Agricultural best practices
* 🇳🇬 English and Hausa language support

---

# ✨ Key Features

* ✅ Fully Offline AI Assistant
* ✅ English & Hausa Support
* ✅ Retrieval-Augmented Generation (RAG)
* ✅ Source-backed Answers
* ✅ Follow-up Conversation Memory
* ✅ Fast Semantic Search using FAISS
* ✅ Lightweight Local Deployment
* ✅ Designed for Rural Communities
* ✅ Runs on Standard 8GB Laptops
* ✅ Local Web Interface
* ✅ Command Line Interface

---

# 🏗 System Architecture

```text
Farmer Question
        │
        ▼
Web UI / CLI
        │
        ▼
Language Detection
        │
        ▼
Conversation Memory
        │
        ▼
Retriever (FAISS)
        │
        ▼
Agricultural Knowledge Base
        │
        ▼
Ollama + Qwen2.5:1.5B
        │
        ▼
Grounded AI Response
```

---

# 🧠 How It Works

1. The farmer asks a question in English or Hausa.
2. AgriSec detects the language.
3. Relevant agricultural documents are retrieved using FAISS.
4. The retrieved knowledge is supplied to the local AI model.
5. The AI generates a grounded response.
6. The system displays the information together with the document sources.

Everything runs locally without any cloud dependency.

---

# 🌍 Example Questions

### English

* How do I start maize farming?
* Which fertilizer is best for rice?
* How can I control Fall Armyworm?
* How should I store maize after harvest?

### Hausa

* Yaya zan shuka masara?
* Wane taki ya dace da shinkafa?
* Yaya zan kashe tsutsar Fall Armyworm?
* Ta yaya zan adana masara bayan girbi?

---

# ⚙ Technologies Used

## Programming

* Python

## AI

* Ollama
* Qwen2.5:1.5B
* Retrieval-Augmented Generation (RAG)

## Retrieval

* FAISS
* Sentence Transformers

## Frontend

* HTML
* CSS
* JavaScript

## Data

* Markdown Knowledge Base

---

# 📁 Project Structure

```text
AgriSec/
│
├── assets/
│
├── data/
│   ├── corpus/
│   └── index/
│
├── reports/
│
├── scripts/
│
├── src/
│   ├── assistant.py
│   ├── inference.py
│   ├── rag.py
│   └── web_app.py
│
├── requirements.txt
├── README.md
├── ARCHITECTURE.md
└── PLAN.md
```

---

# 🚀 Installation

```bash
git clone https://github.com/Ambatogusau/AgriSec.git

cd AgriSec

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

ollama pull qwen2.5:1.5b

python -m src.rag --build
```

---

# ▶ Running AgriSec

Start the Web Application

```bash
python -m src.web_app --model qwen2.5:1.5b
```

Open:

```
http://127.0.0.1:7860
```

Run the Command Line Assistant

```bash
python -m src.assistant --model qwen2.5:1.5b
```

---

# 📸 Screenshots

Screenshots will be added after the final prototype demonstration.

---

# 🎥 Demo Video

Demo video will be added after the final presentation.

---

# 🚀 Future Improvements

* Voice interaction
* Drone integration
* Soil analysis
* Plant disease detection
* Weather forecasting
* Farm mapping
* Mobile application
* IoT sensor integration

---

# 👨‍💻 Developed By

**Abdullahi Badamasi**

Founder & CEO, Ambato Digital Hub

Africa Deep Tech Challenge (ADTC) 2026

---

# 📄 License

This project was developed for the **Africa Deep Tech Challenge (ADTC) 2026**.

For educational and demonstration purposes.
