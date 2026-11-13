"""Regression guards for the module document visibility policy.

Full API/database integration coverage is required before deployment.
"""
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "app" / "api" / "modules.py"


def test_class_publication_requires_module_ownership():
    source = SOURCE.read_text()
    assert 'visibility not in ("class", "personal")' in source
    assert 'visibility == "class" and module.owner_id != current_user.id' in source
    assert 'current_user.role != "admin"' in source
    assert 'current_user.role == "student"' not in source[source.index('async def _ingest_module_document'):source.index('    # Resolve week metadata',source.index('async def _ingest_module_document'))]


def test_personal_notes_are_private_for_all_roles():
    source = SOURCE.read_text()
    start = source.index('async def list_module_documents(')
    end = source.index('    result = await db.execute(query.order_by', start)
    policy = source[start:end]
    assert 'Document.owner_id == current_user.id' in policy
    assert 'if current_user.role in ("student", "self_learner"):' not in policy
