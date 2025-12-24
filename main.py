import logging
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
    get_file_path,
    load_dataframe,
    validate_filename
)
from utils.analytics import analyze_sales_data
from utils.forecast import forecast_demand

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BizPilot AI Backend")
security = HTTPBearer()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "https://www.yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)


# ========== MODELS ==========
class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    stage: str = "existing"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


# ========== DEPENDENCIES ==========
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user"""
    try:
        token = credentials.credentials
        payload = verify_token(token)

        if not payload:
            logger.warning("Invalid token attempted")
            raise HTTPException(401, "Invalid or expired token")

        user = get_user_by_id(payload.get("user_id"))
        if not user:
            logger.warning(f"User not found for token: {payload.get('user_id')}")
            raise HTTPException(401, "User not found")

        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auth error: {str(e)}")
        raise HTTPException(401, "Invalid or expired token")


# ========== PUBLIC ROUTES ==========
@app.get("/")
def root():
    """Health check"""
    return {"message": "BizPilot AI Backend"}


@app.post("/signup")
def signup(data: UserSignup):
    """Register new user"""
    try:
        user = create_user(
            email=data.email,
            password=data.password,
            full_name=data.full_name,
            stage=data.stage
        )
        token = create_access_token({"user_id": user["id"]})
        logger.info(f"User signed up: {data.email}")

        return {
            "user": user,
            "access_token": token,
            "token_type": "bearer"
        }
    except ValueError as e:
        logger.warning(f"Signup error: {str(e)}")
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        raise HTTPException(500, "Signup failed")


@app.post("/login")
def login(data: UserLogin):
    """Authenticate user"""
    user = authenticate_user(data.email, data.password)

    if not user:
        logger.warning(f"Failed login attempt: {data.email}")
        raise HTTPException(401, "Invalid email or password")

    token = create_access_token({"user_id": user["id"]})
    logger.info(f"User logged in: {data.email}")

    return {
        "user": user,
        "access_token": token,
        "token_type": "bearer"
    }


# ========== PROTECTED ROUTES ==========
@app.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Get current user info"""
    return current_user


@app.post("/upload-sales-data")
async def upload_sales(
        file: UploadFile = File(...),
        current_user: dict = Depends(get_current_user)
):
    """Upload sales data file"""
    try:
        result = await save_uploaded_file(file, current_user["id"])
        logger.info(f"File uploaded by {current_user['id']}: {file.filename}")
        return result
    except Exception as e:
        logger.error(f"Upload error for {current_user['id']}: {str(e)}")
        raise


@app.get("/files")
def list_files(current_user: dict = Depends(get_current_user)):
    """List all user files"""
    try:
        files = get_user_files(current_user["id"])
        return {"files": files}
    except Exception as e:
        logger.error(f"List files error for {current_user['id']}: {str(e)}")
        raise


@app.delete("/files/{filename}")
def delete_file(
        filename: str,
        current_user: dict = Depends(get_current_user)
):
    """Delete a file"""
    try:
        validate_filename(filename)
        blob_name = get_file_path(filename, current_user["id"])
        result = delete_user_file(blob_name, current_user["id"])
        logger.info(f"File deleted by {current_user['id']}: {filename}")
        return result
    except Exception as e:
        logger.error(f"Delete error for {current_user['id']}: {str(e)}")
        raise


@app.get("/analytics/{filename}")
def get_analytics(
        filename: str,
        current_user: dict = Depends(get_current_user)
):
    """Get analytics for a file"""
    try:
        validate_filename(filename)
        blob_name = get_file_path(filename, current_user["id"])
        df = load_dataframe(blob_name, current_user["id"])
        analytics = analyze_sales_data(df)
        logger.info(f"Analytics generated for {current_user['id']}: {filename}")
        return analytics
    except Exception as e:
        logger.error(f"Analytics error for {current_user['id']}: {str(e)}")
        raise


class ForecastRequest(BaseModel):
    filename: str
    periods: int = 30


@app.post("/forecast")
def get_forecast(
        request: ForecastRequest,
        current_user: dict = Depends(get_current_user)
):
    """Generate demand forecast for a file"""
    try:
        validate_filename(request.filename)
        blob_name = get_file_path(request.filename, current_user["id"])
        df = load_dataframe(blob_name, current_user["id"])
        forecast = forecast_demand(df, periods=request.periods)
        logger.info(f"Forecast generated for {current_user['id']}: {request.filename}")
        return forecast
    except Exception as e:
        logger.error(f"Forecast error for {current_user['id']}: {str(e)}")
        raise


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)