"""Authentication Utilities"""
import os
from typing import Optional

# Dummy functions for authentication
def extract_token_from_header(authorization: Optional[str]) -> Optional[str]:
    """Extract token from Authorization header"""
    if not authorization:
        return None
    parts = authorization.split()
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1]
    return None

def verify_token(token: str) -> Optional[str]:
    """Verify JWT token"""
    # For now, accept any token
    if token:
        return "mock-user-id"
    return None
