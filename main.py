import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
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
from utils.llm import call_llm_simple, call_llm_with_functions
from utils.research import do_market_research_cached, SearchManager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="BizPilot AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# Initialize search manager for research (shared instance)
try:
    search_manager = SearchManager()
    logger.info("Search manager initialized successfully")
except ValueError as e:
    logger.warning(f"Search manager initialization failed: {str(e)}. Market research will be unavailable.")
    search_manager = None


class UserSignup(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    stage: str = "existing"


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class ForecastRequest(BaseModel):
    filename: str
    periods: int = 30


class LLMRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    use_reasoning: bool = False


class DataQuestionRequest(BaseModel):
    filename: str
    question: str
    model: Optional[str] = None
    use_reasoning: bool = False


def get_token_from_header(authorization: Optional[str] = Header(None)) -> str:
    """Extract token from Authorization header"""
    if not authorization:
        raise HTTPException(401, "Missing authorization header")

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(401, "Invalid authorization header format. Use 'Bearer <token>'")

    return parts[1]


async def get_current_user(token: str = Depends(get_token_from_header)) -> dict:
    """Verify JWT token and return user"""
    try:
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


async def get_current_user_optional(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Get current user if token provided, otherwise return None"""
    if not authorization:
        return None

    try:
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        token = parts[1]
        payload = verify_token(token)

        if not payload:
            return None

        user = get_user_by_id(payload.get("user_id"))
        return user
    except Exception as e:
        logger.warning(f"Optional auth error: {str(e)}")
        return None


def execute_function(function_name: str, arguments: dict) -> dict:
    """Execute business logic functions called by LLM"""
    try:
        if function_name == "list_available_files":
            user_id = arguments["user_id"]
            files = get_user_files(user_id)
            return {
                "status": "success",
                "total_files": len(files),
                "files": files
            }

        elif function_name == "analyze_sales_file":
            filename = arguments["filename"]
            user_id = arguments["user_id"]

            validate_filename(filename)
            blob_name = get_file_path(filename, user_id)
            df = load_dataframe(blob_name, user_id)
            analytics = analyze_sales_data(df)
            return analytics

        elif function_name == "query_sales_data":
            filename = arguments["filename"]
            user_id = arguments["user_id"]
            question = arguments["question"]

            validate_filename(filename)
            blob_name = get_file_path(filename, user_id)
            df = load_dataframe(blob_name, user_id)
            analytics = analyze_sales_data(df)

            return {
                "question": question,
                "data_summary": analytics,
                "instruction": "Analyze the data_summary and provide a clear answer to the user's question with specific numbers and insights."
            }

        elif function_name == "forecast_sales_demand":
            filename = arguments["filename"]
            user_id = arguments["user_id"]
            periods = arguments.get("periods", 30)

            if periods < 1 or periods > 365:
                return {"status": "error", "message": "Periods must be between 1 and 365"}

            validate_filename(filename)
            blob_name = get_file_path(filename, user_id)
            df = load_dataframe(blob_name, user_id)
            result = forecast_demand(df, periods)
            return result

        elif function_name == "market_research":
            if not search_manager:
                return {
                    "status": "error",
                    "message": "Market research is unavailable. Google API credentials not configured."
                }

            idea = arguments.get("idea", "")
            customer = arguments.get("customer", "")
            geography = arguments.get("geography", "")
            level = arguments.get("level", 1)

            if not all([idea, customer, geography]):
                return {
                    "status": "error",
                    "message": "Market research requires: idea, customer, and geography"
                }

            try:
                result = do_market_research_cached(
                    idea=idea,
                    customer=customer,
                    geography=geography,
                    level=min(level, 3),  # Cap at level 3
                    search_manager=search_manager,
                    cache_dir="./cache",
                    cache_expiry_hours=24
                )
                return {
                    "status": "success",
                    "level": result["level"],
                    "idea": result["idea"],
                    "research": result.get("research", ""),
                    "searches_performed": result.get("searches_successful", 0),
                    "usage_stats": result.get("usage_stats", {}),
                    "note": "Results may be cached from previous searches"
                }
            except Exception as e:
                logger.error(f"Market research error: {str(e)}")
                return {
                    "status": "error",
                    "message": f"Research failed: {str(e)}"
                }

        else:
            return {"status": "error", "message": f"Unknown function: {function_name}"}

    except Exception as e:
        logger.error(f"Function execution error in {function_name}: {str(e)}")
        return {"status": "error", "message": str(e)}


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


@app.post("/llm/chat")
def llm_chat(
        request: LLMRequest,
        current_user: Optional[dict] = Depends(get_current_user_optional)
):
    """Simple chat with LLM (no function calling, authentication optional)"""
    try:
        user_id = current_user["id"] if current_user else "anonymous"
        response = call_llm_simple(
            prompt=request.prompt,
            model=request.model,
            use_reasoning=request.use_reasoning
        )
        logger.info(f"LLM chat for {user_id} (reasoning: {request.use_reasoning})")
        return {"response": response}
    except Exception as e:
        user_id = current_user["id"] if current_user else "anonymous"
        logger.error(f"LLM error for {user_id}: {str(e)}")
        raise HTTPException(500, f"LLM call failed: {str(e)}")


@app.post("/llm/general-assistant")
def general_assistant(
        request: LLMRequest,
        current_user: dict = Depends(get_current_user)
):
    """General business assistant with access to all your data, analytics, and research functions"""
    try:
        user_id = current_user["id"]

        # Get list of user's files
        user_files = get_user_files(user_id)
        files_list = ", ".join([f["original_filename"] for f in user_files]) if user_files else "No files uploaded"

        # Check research availability
        research_note = ""
        if search_manager:
            stats = search_manager.get_usage_stats()
            research_note = f"\n\nResearch Tool Available: Google API quota {stats['used']}/{stats['limit']} used today. You can research market ideas, competitors, and customer segments."
        else:
            research_note = "\n\nNote: Market research tool is unavailable (Google API not configured)."

        prompt = f"""You are BizPilot AI, a business intelligence assistant.

User's Question: {request.prompt}

Available Files: {files_list}

Available Functions:
- analyze_sales_file: Deep dive analysis of uploaded sales data
- query_sales_data: Answer specific questions about sales data
- forecast_sales_demand: Predict future sales trends
- list_available_files: See all user files
- market_research: Research markets, competitors, and customer segments{research_note}

Instructions:
1. If the user asks about their data, use the sales analysis functions
2. If the user asks about markets, competition, or business ideas, use market_research
3. Otherwise, answer directly with your knowledge
4. Be professional and data-driven
5. Do NOT ask for filenames or user IDs - use the functions to find them
6. When using market_research, choose level: 1 (quick), 2 (medium), or 3 (deep)"""

        result = call_llm_with_functions(
            prompt=prompt,
            function_executor=execute_function,
            context={"user_id": user_id},
            model=request.model,
            use_reasoning=request.use_reasoning,
            max_iterations=5
        )

        logger.info(f"General assistant query for {user_id} (reasoning: {request.use_reasoning})")
        return {
            "response": result["response"],
            "reasoning_details": result.get("reasoning_details")
        }
    except Exception as e:
        logger.error(f"General assistant error for {current_user['id']}: {str(e)}")
        raise HTTPException(500, f"Assistant call failed: {str(e)}")


@app.get("/research/usage")
def get_research_usage():
    """Get current market research API usage stats"""
    if not search_manager:
        return {"status": "unavailable", "message": "Google API not configured"}

    stats = search_manager.get_usage_stats()
    return {"status": "available", "usage": stats}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)