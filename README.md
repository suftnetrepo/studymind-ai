# StudyMind AI — Backend API

AI-powered study assistant platform. Students upload course materials and get
grounded answers, quizzes, flashcards, and summaries powered by GPT-4o and
LlamaIndex with Typesense hybrid search.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.12+ |
| AI Framework | LlamaIndex 0.14.x |
| LLM | OpenAI GPT-4o |
| Embeddings | text-embedding-3-large (3072 dim) |
| Vector Search | Typesense 0.25.2 |
| API | FastAPI + Uvicorn |
| Database | PostgreSQL 14+ via SQLAlchemy 2.0 async |
| Validation | Pydantic v2 |
| Auth | JWT (python-jose) + bcrypt (passlib) |
| Testing | pytest — 195 tests, 100% passing |

---

## Quick Start

### 1. Prerequisites

```bash
# macOS
brew install python@3.12 postgresql@14 colima docker docker-compose

# Start Docker VM (macOS only)
colima start --arch x86_64 --cpu 4 --memory 8
```

### 2. Clone and install

```bash
cd studymind-final
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env — add your OPENAI_API_KEY
```

### 4. Start Typesense

```bash
docker compose up -d
curl http://localhost:8108/health   # → {"ok":true}
```

### 5. Create database

```bash
psql -U postgres -c "CREATE DATABASE studymind;"
python scripts/create_db.py
```

### 6. Seed demo data (optional)

```bash
python scripts/seed_demo.py
```

This creates demo accounts and a sample module so you can test immediately.

### 7. Run the API

```bash
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 8. Open docs

```
http://localhost:8000/docs
```

### 9. Run tests

```bash
pytest tests/ -v
# Expected: 195 passed
```

---

## Testing the API manually

### Step 1 — Register or use demo credentials

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "lecturer@demo.ac.uk", "password": "Lecturer1234"}'
```

Copy the `access_token` from the response.

### Step 2 — Authorise in Swagger

Open http://localhost:8000/docs → click **Authorize** → paste:
```
Bearer <your_access_token>
```

### Step 3 — Upload a document

```bash
curl -X POST http://localhost:8000/api/modules/{module_id}/documents \
  -H "Authorization: Bearer <token>" \
  -F "file=@your_document.pdf" \
  -F "visibility=class"
```

### Step 4 — Ask a question

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is a pointer?",
    "module_id": "<module_id>",
    "scope_mode": "everything"
  }'
```

### Step 5 — Generate a quiz

```bash
curl -X POST http://localhost:8000/api/quiz/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "module_id": "<module_id>",
    "question_count": 5,
    "question_type": "mcq"
  }'
```

### Step 6 — Generate flashcards

```bash
curl -X POST http://localhost:8000/api/flashcards/generate \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"module_id": "<module_id>", "max_cards": 10}'
```

### Step 7 — Summarise a module

```bash
curl -X POST http://localhost:8000/api/summarise \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"module_id": "<module_id>", "scope": "module"}'
```

---

## Scope shortcuts in chat

Prefix your message with a course code to scope the search instantly:

```json
{ "message": "/csc109 What is recursion?" }
```

Or scope to a specific week:

```json
{ "message": "/csc109/week3 Explain binary trees" }
```

---

## Project Structure

```
app/
├── agents/
│   ├── llm_factory.py          # OpenAI LLM + embedding model
│   ├── rag_pipeline.py         # Scoped RAG Q&A pipeline
│   └── ai_features.py          # Quiz, flashcard, summary generation
├── api/
│   ├── main.py                 # FastAPI app factory
│   ├── auth.py                 # JWT auth endpoints
│   ├── institutions.py         # Institution & user management
│   ├── modules.py              # Modules, weeks, enrolments, uploads
│   ├── chat.py                 # Scoped Q&A chat endpoints
│   ├── features.py             # Quiz, flashcard, summariser endpoints
│   ├── documents.py            # Legacy document endpoints
│   └── health.py               # Health + stats
├── auth/
│   ├── security.py             # JWT + bcrypt
│   └── dependencies.py         # require_auth, require_admin etc.
├── db/
│   ├── engine.py               # Async SQLAlchemy engine
│   ├── models.py               # All ORM models (16 tables)
│   └── schemas.py              # Pydantic v2 schemas
├── ingestion/
│   └── ingestor.py             # Document ingestion pipeline
└── retrieval/
    ├── retriever.py            # ScopedHybridRetriever + SearchScope
    └── typesense_client.py     # Typesense operations
scripts/
├── create_db.py                # Create all tables
└── seed_demo.py                # Seed demo data
tests/
├── test_auth.py                # 21 auth tests
├── test_institutions.py        # 33 institution tests
├── test_modules.py             # 40 module tests
├── test_sprint4.py             # 40 scoped Q&A tests
├── test_sprint5.py             # 38 quiz/flashcard/summary tests
├── test_ingestion.py           # 6 ingestion tests
├── test_retriever.py           # 8 retriever tests
└── test_schemas.py             # 9 schema tests
```

---

## Sprints completed

| Sprint | Feature | Tests |
|--------|---------|-------|
| 1 | Auth & JWT | 21 |
| 2 | Institution & User Management | 33 |
| 3 | Modules, Semesters & Document Upload | 40 |
| 4 | Scoped RAG Q&A with Citations | 40 |
| 5 | Quiz, Flashcards & Summariser | 38 + 23 |
| — | Ingestion & Retrieval | 14 |
| **Total** | | **195** |

---

## Next steps

- Sprint 6: React Native mobile app (iOS + Android)
- Sprint 7: React Native AI feature screens
- Sprint 8: Analytics & admin dashboard
- Sprint 9: Personal notes, gap finder, note comparison
- Sprint 10: Security hardening, CI/CD, production deployment

# studymind-ai
