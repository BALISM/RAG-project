# Contributing to RAG Chatbot

Thank you for considering contributing to RAG Chatbot!

## Development Setup

1. Clone the repository and create a Python 3.11 virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\Activate.ps1
   ```
2. Install runtime and development dependencies:
   ```bash
   pip install -r requirements.txt -r requirements-dev.txt
   ```
3. Copy `.env.example` to `.env` and configure your API key.

## Code Quality & Testing

Before submitting a Pull Request, ensure all tests pass and code style guidelines are met:

```bash
# Run pytest test suite
pytest test/ -v

# Run linting with Ruff
ruff check app/ test/

# Run type checking
mypy app/
```

## Pull Request Guidelines

- Create a feature branch off `main`.
- Write unit tests for new functionality or bug fixes.
- Keep commits focused and provide descriptive commit messages.
