"""Authentication for trusted Streamlit-to-API requests."""

import os
from pathlib import Path
import secrets

from dotenv import load_dotenv
from fastapi import Header, HTTPException


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INTERNAL_TOKEN_ENV = "SMARTDOCS_INTERNAL_AUTH_TOKEN"


def internal_auth_token() -> str | None:
    """Read the server-to-server key from the environment or local .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv(INTERNAL_TOKEN_ENV)


def require_streamlit_user(
    x_smartdocs_internal_token: str | None = Header(default=None),
    x_smartdocs_user_id: str | None = Header(default=None),
) -> str:
    """Accept identity only from the authenticated Streamlit server."""
    expected_token = internal_auth_token()
    if not expected_token:
        raise HTTPException(
            status_code=503,
            detail=f"Server authentication is not configured ({INTERNAL_TOKEN_ENV}).",
        )
    if not x_smartdocs_internal_token or not secrets.compare_digest(
        x_smartdocs_internal_token, expected_token
    ):
        raise HTTPException(status_code=401, detail="Authentication required.")
    if not x_smartdocs_user_id or len(x_smartdocs_user_id) > 128:
        raise HTTPException(status_code=401, detail="Authenticated user is missing.")
    return x_smartdocs_user_id
