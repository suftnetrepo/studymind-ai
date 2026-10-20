#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Quanta + StudyMind — local dev setup script
# Run this ONCE from the studymind-ai directory: bash setup-local.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

BOLD='\033[1m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}✓${NC} $1"; }
info() { echo -e "${CYAN}→${NC} $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
hdr()  { echo -e "\n${BOLD}$1${NC}"; }

# ── 0. Pre-flight ─────────────────────────────────────────────────────────────
hdr "0. Checking prerequisites"

command -v python3  >/dev/null || fail "python3 not found — install Python 3.11+"
command -v pip3     >/dev/null || fail "pip3 not found"
command -v psql     >/dev/null || fail "psql not found — install PostgreSQL 14+"
command -v docker   >/dev/null || fail "docker not found — install Docker Desktop"
command -v node     >/dev/null || fail "node not found — install Node.js 20+"
command -v npm      >/dev/null || fail "npm not found"

PYTHON_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
info "Python $PYTHON_VER · Node $(node --version) · npm $(npm --version)"
ok "All prerequisites found"

# ── 1. Typesense via Docker ───────────────────────────────────────────────────
hdr "1. Starting Typesense (Docker)"

if docker ps --format '{{.Names}}' | grep -q "studymind_typesense"; then
  ok "Typesense container already running"
else
  docker rm -f studymind_typesense 2>/dev/null || true
  docker run -d \
    --name studymind_typesense \
    --restart unless-stopped \
    -p 8108:8108 \
    -v typesense_data:/data \
    typesense/typesense:0.25.2 \
    --data-dir /data \
    --api-key studymind-typesense-key \
    --enable-cors
  info "Waiting for Typesense to be ready..."
  for i in $(seq 1 15); do
    if curl -sf http://localhost:8108/health >/dev/null 2>&1; then
      ok "Typesense ready on :8108"; break
    fi
    sleep 1
  done
fi

# ── 2. PostgreSQL databases ───────────────────────────────────────────────────
hdr "2. Setting up PostgreSQL databases"

PG_USER=${PGUSER:-postgres}
PG_HOST=${PGHOST:-localhost}
PG_PORT=${PGPORT:-5432}

pg_isready -h $PG_HOST -p $PG_PORT -U $PG_USER >/dev/null 2>&1 || \
  fail "PostgreSQL is not running on $PG_HOST:$PG_PORT. Start it first."

run_psql() { psql -h $PG_HOST -p $PG_PORT -U $PG_USER -c "$1" 2>/dev/null || true; }

run_psql "CREATE USER quanta WITH PASSWORD 'quanta' CREATEDB;"
run_psql "CREATE DATABASE studymind OWNER quanta;"
run_psql "CREATE DATABASE quanta    OWNER quanta;"
run_psql "GRANT ALL PRIVILEGES ON DATABASE studymind TO quanta;"
run_psql "GRANT ALL PRIVILEGES ON DATABASE quanta    TO quanta;"
ok "Databases 'studymind' and 'quanta' ready"

# ── 3. StudyMind .env ─────────────────────────────────────────────────────────
hdr "3. StudyMind .env"

if [ ! -f .env ]; then
  cp .env.example .env
  info "Created .env from .env.example"
  echo ""
  echo -e "${RED}  !! ACTION REQUIRED: open studymind-ai/.env and set OPENAI_API_KEY !!${NC}"
  echo ""
else
  ok ".env already exists — skipping"
fi

# ── 4. Python virtual environment ─────────────────────────────────────────────
hdr "4. Python dependencies"

if [ ! -d .venv ]; then
  python3 -m venv .venv
  ok "Created .venv"
fi

source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q
ok "Python packages installed"

# ── 5. Create DB tables ───────────────────────────────────────────────────────
hdr "5. Creating StudyMind database tables"

python3 scripts/create_db.py && ok "Tables created" || \
  info "create_db.py failed — tables may already exist or OPENAI_API_KEY not set yet (harmless)"

# ── 6. Quanta .env.local ──────────────────────────────────────────────────────
hdr "6. Quanta Next.js .env.local"

QUANTA_DIR="$(dirname "$0")/../quanta"
if [ -d "$QUANTA_DIR" ]; then
  ENV_FILE="$QUANTA_DIR/.env.local"
  if [ ! -f "$ENV_FILE" ]; then
    cat > "$ENV_FILE" <<'ENVEOF'
# Quanta — local development
DATABASE_URL="postgresql://quanta:quanta@localhost:5432/quanta"

# NextAuth v5
NEXTAUTH_SECRET="quanta-dev-secret-change-in-production"
NEXTAUTH_URL="http://localhost:3000"

# OpenAI (for voice sessions)
OPENAI_API_KEY=""

# StudyMind AI backend (NeuralCore)
NEURALCORE_API_URL="http://localhost:8000"
NEURALCORE_API_KEY=""

# Cloudinary (optional for local — can leave blank)
CLOUDINARY_CLOUD_NAME=""
CLOUDINARY_API_KEY=""
CLOUDINARY_API_SECRET=""

NEXT_PUBLIC_APP_URL="http://localhost:3000"
NEXT_PUBLIC_APP_NAME="Quanta"
NEXT_PUBLIC_NEURALCORE_URL="http://localhost:8000"
ENVEOF
    ok "Created quanta/.env.local"
  else
    ok "quanta/.env.local already exists — skipping"
  fi
else
  info "quanta/ directory not found at $QUANTA_DIR — skipping (run from studymind-ai/)"
fi

# ── 7. Quanta npm install ─────────────────────────────────────────────────────
hdr "7. Quanta npm install"

if [ -d "$QUANTA_DIR" ]; then
  (cd "$QUANTA_DIR" && npm install --silent) && ok "Quanta npm packages installed"
else
  info "quanta/ not found — skipping npm install"
fi

# ── 8. Drizzle push (Quanta DB schema) ────────────────────────────────────────
hdr "8. Quanta database schema (Drizzle)"

if [ -d "$QUANTA_DIR" ]; then
  (cd "$QUANTA_DIR" && npm run drizzle:push 2>&1) && ok "Quanta schema pushed to DB" || \
    info "drizzle:push failed — OPENAI_API_KEY may not be needed but check DATABASE_URL"
else
  info "quanta/ not found — skipping drizzle push"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}${BOLD} Setup complete!${NC}"
echo -e "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Next steps:"
echo "  1. Set OPENAI_API_KEY in studymind-ai/.env"
echo "  2. Set OPENAI_API_KEY in quanta/.env.local (for voice)"
echo ""
echo "  Start the stack (3 terminals):"
echo ""
echo -e "  ${CYAN}Terminal 1 — StudyMind AI (FastAPI):${NC}"
echo "    cd studymind-ai"
echo "    source .venv/bin/activate"
echo "    uvicorn app.api.main:app --reload --port 8000"
echo ""
echo -e "  ${CYAN}Terminal 2 — Quanta (Next.js):${NC}"
echo "    cd quanta"
echo "    npm run dev"
echo ""
echo -e "  ${CYAN}URLs:${NC}"
echo "    Quanta UI      → http://localhost:3000"
echo "    StudyMind API  → http://localhost:8000/docs"
echo "    Typesense      → http://localhost:8108/health"
echo ""
