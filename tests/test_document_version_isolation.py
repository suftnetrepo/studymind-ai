"""Static regression guards; run full API/Postgres/Typesense integration before staging."""
from pathlib import Path

SOURCE = (Path(__file__).resolve().parents[1] / "app/api/modules.py").read_text()


def test_version_lookup_scopes_filename_owner_and_visibility():
    start = SOURCE.index('    # Determine version — MOD-04')
    end = SOURCE.index('    # Create Document record', start)
    body = SOURCE[start:end]
    assert 'Document.filename == filename' in body
    assert 'Document.owner_id == current_user.id' in body
    assert 'ModuleDocument.visibility == visibility' in body
    assert 'ModuleDocument.is_latest == True' in body


def test_version_history_requires_same_owner_and_visibility():
    start = SOURCE.index('async def get_document_versions(')
    end = SOURCE.index('\n\n# ═', start)
    body = SOURCE[start:end]
    assert 'Document.id == document_id' in body
    assert 'target_md.visibility == "personal" and target_doc.owner_id != current_user.id' in body
    assert 'Document.owner_id == target_doc.owner_id' in body
    assert 'ModuleDocument.visibility == target_md.visibility' in body


def test_module_owner_cannot_delete_another_personal_note():
    start = SOURCE.index('async def delete_module_document(')
    end = SOURCE.index('@router.get(', start)
    body = SOURCE[start:end]
    assert 'md.visibility == "class"' in body
    assert 'doc.owner_id != current_user.id' in body
