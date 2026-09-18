"""
One-time script: create all PostgreSQL tables for StudyMind AI.
Run this once after setting up your database.

Usage:
    python scripts/create_db.py
"""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

async def main():
    from app.db.engine import create_all_tables
    from app.logging_config import configure_logging
    configure_logging()
    print("Creating all tables...")
    await create_all_tables()
    print("✅ All tables created successfully.")

if __name__ == "__main__":
    asyncio.run(main())
