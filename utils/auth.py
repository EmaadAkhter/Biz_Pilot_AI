import os
from datetime import datetime, timedelta
from jose import JWTError, jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from dotenv import load_dotenv
from supabase import create_client, Client
import hashlib

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ph = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash password using Argon2"""
    return ph.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    try:
        ph.verify(hashed_password, plain_password)
        return True
    except VerifyMismatchError:
        return False


def hash_string(text: str) -> str:
    """Create SHA256 hash of string for filename prefixes"""
    return hashlib.sha256(text.encode()).hexdigest()


def create_access_token(data: dict) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> dict:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None


def create_user(email: str, password: str, full_name: str = None, stage: str = "existing") -> dict:
    """Create new user in Supabase"""
    response = supabase.table("users").select("*").eq("email", email).execute()

    if response.data:
        raise ValueError("Email already exists")

    hashed_pw = hash_password(password)

    user_data = {
        "email": email,
        "hashed_password": hashed_pw,
        "full_name": full_name,
        "stage": stage,
        "created_at": datetime.utcnow().isoformat()
    }

    response = supabase.table("users").insert(user_data).execute()

    if not response.data:
        raise Exception("Failed to create user")

    user = response.data[0]
    del user["hashed_password"]
    return user


def authenticate_user(email: str, password: str) -> dict:
    """Authenticate user by email and password"""
    response = supabase.table("users").select("*").eq("email", email).execute()

    if not response.data:
        return None

    user = response.data[0]

    if not verify_password(password, user["hashed_password"]):
        return None

    del user["hashed_password"]
    return user


def get_user_by_id(user_id: str) -> dict:
    """Get user by ID from Supabase"""
    response = supabase.table("users").select("*").eq("id", user_id).execute()

    if not response.data:
        return None

    user = response.data[0]
    del user["hashed_password"]
    return user
