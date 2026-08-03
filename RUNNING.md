# 🛠️ How to Run RAG Chatbot

Quick reference guide for development, running locally, Docker execution, and testing.

---

## ⚡ First-Time Local Setup

```powershell
cd D:\rag-chatbot

# Create virtualenv with Python 3.11+
py -3.11 -m venv venv
venv\Scripts\Activate.ps1

# Install runtime & dev packages
pip install -r requirements.txt -r requirements-dev.txt

# Create .env from template
copy .env.example .env
```

Open `.env` and set your `GEMINI_API_KEY`.

---

## 🏃 Running the Server

```powershell
venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Open your browser at **http://127.0.0.1:8000**.  
Interactive API Docs (Swagger): **http://127.0.0.1:8000/docs**

---

## 🐳 Running with Docker

```powershell
# Build and start container in background
docker-compose up --build -d

# Check logs
docker-compose logs -f
```

---

## 🧪 Executing Automated Tests

```powershell
pytest test/ -v
```

---

## 🧹 Total Clean Slate Reset

To wipe all uploaded files, vector embeddings, and conversation histories:

```powershell
Remove-Item -Recurse -Force chroma_db, uploads, chat_sessions.db
```