"""StudyMind Web-only Checkout endpoint; never activates access from the browser."""
from fastapi import APIRouter, Depends, HTTPException
from app.auth.dependencies import require_auth
from app.db.models import User
from app.billing.stripe_checkout import checkout_configured, create_checkout
from app.logging_config import get_logger

router = APIRouter(prefix="/api/billing", tags=["billing"])
log = get_logger(__name__)


@router.post("/checkout")
async def checkout(current_user: User = Depends(require_auth)):
    if not checkout_configured():
        raise HTTPException(status_code=503, detail="StudyMind Plus checkout is not available yet.")
    try:
        return await create_checkout(str(current_user.id), current_user.email)
    except Exception as exc:
        log.error("studymind_checkout_failed", error_type=type(exc).__name__)
        raise HTTPException(status_code=502, detail="Unable to start checkout. Please try again.") from None
