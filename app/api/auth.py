"""
Authentication API Routes
FIXED: Now creates real users in database and generates real JWT tokens
WITH: Password hashing, token verification, and /api/auth/me endpoint
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from uuid import uuid4
import jwt
from passlib.context import CryptContext
from typing import Optional
from app.core.database import get_db
from app.models.models import User
from app.schemas import UserRegister, UserLogin, TokenResponse, UserResponse

router = APIRouter(prefix="", tags=["Authentication"])

# JWT config
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(user_id: str):
    """Create JWT token"""
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "user_id": user_id,
        "exp": expire
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    return token

def verify_token(token: str) -> dict:
    """Verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return {"user_id": user_id}
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# ============================================================================
# DEPENDENCY: GET CURRENT USER (NEW - FIXES /api/auth/me)
# ============================================================================

async def get_current_user(
        db: Session = Depends(get_db),
        authorization: Optional[str] = None,
) -> User:
    """
    Get current authenticated user from token
    Used by /api/auth/me endpoint
    """

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication scheme"
            )
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header"
        )

    # Verify token
    token_data = verify_token(token)
    user_id = token_data.get("user_id")

    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )

    return user

# ============================================================================
# ENDPOINTS (KEEP EXISTING + ADD NEW)
# ============================================================================

@router.post("/register", response_model=TokenResponse)
async def register(user: UserRegister, db: Session = Depends(get_db)):
    """Register a new user - creates in database

    ✅ EXISTING ENDPOINT - KEPT INTACT
    """

    try:
        # Check if user already exists by username
        existing_user = db.query(User).filter(User.username == user.username).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already registered"
            )

        # Check if email already exists
        existing_email = db.query(User).filter(User.email == user.email).first()
        if existing_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )

        # ✅ FIX: Hash password instead of storing plain text
        hashed_password = hash_password(user.password)

        # Create new user
        new_user = User(
            id=str(uuid4()),
            username=user.username,
            email=user.email,
            password_hash=hashed_password,  # ✅ FIXED: Use hashed password
            created_at=datetime.now()
        )
        db.add(new_user)
        db.commit()
        db.refresh(new_user)

        # Generate token
        token = create_access_token(new_user.id)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": new_user.id,
                "username": new_user.username,
                "email": new_user.email,
                "created_at": new_user.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Registration error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin, db: Session = Depends(get_db)):
    """Login user - FIXED to require existing user with password verification

    ✅ EXISTING ENDPOINT - IMPROVED
    """

    try:
        # Try to find user by username first
        db_user = db.query(User).filter(User.username == user.username).first()

        # If not found by username, try by email
        if not db_user and "@" in user.username:
            db_user = db.query(User).filter(User.email == user.username).first()

        # ✅ FIX: User must exist (don't auto-create)
        if not db_user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # ✅ FIX: Verify password (not just use it as-is)
        if not verify_password(user.password, db_user.password_hash):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password"
            )

        # Generate token
        token = create_access_token(db_user.id)

        return {
            "access_token": token,
            "token_type": "bearer",
            "user": {
                "id": db_user.id,
                "username": db_user.username,
                "email": db_user.email,
                "created_at": db_user.created_at.isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login failed"
        )

# ============================================================================
# 🆕 NEW ENDPOINTS (FIXES FRONTEND ERRORS)
# ============================================================================

@router.get("/me", response_model=UserResponse)
async def get_current_user_endpoint(
        current_user: User = Depends(get_current_user),
) -> dict:
    """
    🆕 NEW ENDPOINT: Get current authenticated user

    This is what your frontend calls!
    Requires: Authorization: Bearer <token>

    Returns: User info

    ✅ FIXES: 404 error on /api/auth/me
    """

    return {
        "id": current_user.id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at.isoformat()
    }

@router.post("/refresh", response_model=TokenResponse)
async def refresh(
        current_user: User = Depends(get_current_user),
) -> dict:
    """
    🆕 IMPROVED ENDPOINT: Refresh token

    Now properly validates the user and refreshes their token
    """

    new_token = create_access_token(current_user.id)

    return {
        "access_token": new_token,
        "token_type": "bearer",
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "email": current_user.email,
            "created_at": current_user.created_at.isoformat()
        }
    }

@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)) -> dict:
    """
    🆕 NEW ENDPOINT: Logout user

    Client should discard the token after this
    """

    return {
        "message": "Successfully logged out",
        "user_id": current_user.id
    }

@router.get("/users/{user_id}", response_model=UserResponse)
async def get_user(
        user_id: str,
        db: Session = Depends(get_db),
        current_user: User = Depends(get_current_user),
) -> dict:
    """
    🆕 NEW ENDPOINT: Get user by ID

    Requires authentication
    """

    user = db.query(User).filter(User.id == user_id).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    return {
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat()
    }