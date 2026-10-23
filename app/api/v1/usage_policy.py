"""
v1 usage-policy endpoints (additive). See app/usage/policy.py for the enforcement model.

GET  /api/v1/usage/policy         — status of a pass: per-feature allowance used/remaining today,
                                    reset time, expiry. Session token → its own pass (no cost data);
                                    API key → any pass of this integration (includes the budget).
POST /api/v1/usage/policy/revoke  — API key only. Revokes a pass (e.g. refund / dispute); every
                                    token minted under it stops working immediately. Idempotent.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.auth.api_key_auth import PlatformIdentity, get_platform_identity
from app.db.engine import get_db
from app.usage import policy as usage

router = APIRouter(prefix="/v1", tags=["Platform API"])


class RevokePolicyRequest(BaseModel):
    pass_id: str = Field(..., min_length=1, max_length=255)
    user_id: str = Field(..., min_length=1, max_length=255)
    reason:  Optional[str] = Field(default=None, max_length=255)


@router.get("/usage/policy")
async def get_usage_policy(
    pass_id:  Optional[str] = Query(default=None, max_length=255),
    user_id:  Optional[str] = Query(default=None, max_length=255),
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    if identity.auth_type == "session_token":
        if identity.policy is None:
            raise usage.UsagePolicyError(404, "no_policy", "This session has no AI Tutor Pass.")
        if (pass_id and pass_id != identity.policy.external_ref) or (user_id and user_id != identity.user_id):
            raise usage.UsagePolicyError(404, "policy_not_found", "No AI Tutor Pass found.")
        return await usage.policy_status(db, identity.policy, include_budget=False)

    if not pass_id or not user_id:
        raise usage.UsagePolicyError(422, "policy_invalid", "pass_id and user_id are required.")
    policy = await usage.find_policy(db, identity.api_key, pass_id, user_id)
    return await usage.policy_status(db, policy, include_budget=True)


@router.post("/usage/policy/revoke")
async def revoke_usage_policy(
    req:      RevokePolicyRequest,
    db:       AsyncSession = Depends(get_db),
    identity: PlatformIdentity = Depends(get_platform_identity),
):
    if identity.auth_type != "api_key":
        raise usage.UsagePolicyError(403, "forbidden", "Passes can only be revoked server-to-server.")
    policy = await usage.revoke_policy(db, identity.api_key, req.pass_id, req.user_id, req.reason)
    return await usage.policy_status(db, policy, include_budget=True)
