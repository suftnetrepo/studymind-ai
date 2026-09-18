"""Global pytest configuration — stubs required env vars before any import."""
import os

os.environ.setdefault("OPENAI_API_KEY",   "sk-test-key")
os.environ.setdefault("JWT_SECRET_KEY",   "test-secret-key-minimum-32-characters-long")
os.environ.setdefault("POSTGRES_DB",      "studymind_test")
os.environ.setdefault("TYPESENSE_API_KEY","test-key")
