from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional

from utils.auth import (
    create_user,
    authenticate_user,
    create_access_token,
    verify_token,
    get_user_by_id
)
from utils.file_handler import (
    save_uploaded_file,
    get_user_files,
    delete_user_file,
    get_file_path
)
from utils.analytics import analyze_sales_data

app = FastAPI(title="BizPilot AI Backend")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Models
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    stage: str = "existing"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# Dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user data"""
    token = credentials.credentials
    payload = verify_token(token)

    if not payload:
        raise HTTPException(401, "Invalid or expired token")

    user = get_user_by_id(payload.get("user_id"))
    if not user:
        raise HTTPException(401, "User not found")

    return user


# Routes
@app.get("/")
def root():
    """Health check endpoint"""
    return {"message": "BizPilot AI Backend - POC with Auth"}


@app.post("/signup")
def signup(data: UserSignup):
    """Register a new user in Supabase and return JWT token"""
    try:
        user = create_user(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            stage=data.stage
        )

        token = create_access_token({"user_id": user["id"]})

        return {
            "user": user,
            "access_token": token,
            "token_type": "bearer"
        }
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Signup failed: {str(e)}")


@app.post("/login")
def login(data: UserLogin):
    """Authenticate user and return JWT token"""
    user = authenticate_user(data.email, data.password)

    if not user:
        raise HTTPException(401, "Invalid email or password")

    token = create_access_token({"user_id": user["id"]})

    return {
        "user": user,
        "access_token": token,
        "token_type": "bearer"
    }


@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current authenticated user info"""
    return current_user


@app.post("/upload-sales-data")
async def upload_sales(
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user)
):
    """Upload sales data CSV/Excel file with user-specific filename"""
    try:
        result = await save_uploaded_file(file, current_user["id"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")


@app.get("/analytics/{filename}")
def get_analytics(
        filename: str,
        current_user: dict = Depends(get_current_user)
):
    """Get comprehensive analytics for uploaded sales data - optimized for frontend visualization"""
    try:
        filepath = get_file_path(filename, current_user["id"])
        analytics = analyze_sales_data(filepath)
        return analytics
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.get("/files")
def list_files(current_user: dict = Depends(get_current_user)):
    """List all uploaded files for current user"""
    try:
        files = get_user_files(current_user["id"])
        return {"files": files}
    except Exception as e:
        raise HTTPException(500, f"Failed to list files: {str(e)}")


@app.delete("/files/{filename}")
def delete_file(
        filename: str,
        current_user: dict = Depends(get_current_user)
):
    """Delete a user's uploaded file"""
    try:
        result = delete_user_file(filename, current_user["id"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)