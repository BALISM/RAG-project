# How to run this project

Copy-paste reference only. For what the code actually does, see README.md.

---

## First-time setup (only do this once)

```powershell
cd D:\rag-chatbot

py -3.11 -m venv venv
venv\Scripts\Activate.ps1

pip install -r requirements.txt

copy .env.example .env
```

Now open `.env` in VS Code and set your real key:
---

## Every time you want to run it

```powershell
cd D:\rag-chatbot
venv\Scripts\Activate.ps1
uvicorn app.main:app --reload
```

Then open in your browser: **http://127.0.0.1:8000**

To stop the server: `Ctrl+C` in that terminal.

---

## Running the tests

Only needed once, to install test-only tools:
```powershell
pip install -r requirements-dev.txt
```

Then, any time:
```powershell
pytest tests/ -v
```

---

## Quick troubleshooting

**`uvicorn` / `pytest` "not recognized"**
→ Your venv isn't active. Run `venv\Scripts\Activate.ps1` first — your
prompt should show `(venv)` at the start of the line before you run
anything else.

**Made a code change but the server didn't pick it up**
→ `--reload` handles Python file changes automatically. `.env` changes do
**not** auto-reload — stop the server (`Ctrl+C`) and run
`uvicorn app.main:app --reload` again.

**Want a totally clean slate (wipe all uploaded documents + chat history)**
```powershell
# server must be stopped first
Remove-Item -Recurse -Force chroma_db, uploads, chat_sessions.db
```
These get recreated automatically next time you run the server.

**Git commands used throughout this project**
```powershell
git add <file>
git commit -m "message"
git push
```
or to commit everything that changed at once:
```powershell
git commit -am "message"
git push
```