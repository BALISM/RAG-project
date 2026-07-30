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