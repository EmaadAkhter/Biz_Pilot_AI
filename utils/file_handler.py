import uuid
import io
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError, AzureError

# ========== LOGGING ==========
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========== CONFIG ==========
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = "salesdata"
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Validate required environment variables
if not AZURE_ACCOUNT_NAME:
    raise RuntimeError("Missing required environment variable: AZURE_ACCOUNT_NAME")
if not AZURE_SAS_TOKEN:
    raise RuntimeError("Missing required environment variable: AZURE_SAS_TOKEN")

# Fix: Proper Azure Blob Service Client initialization
account_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
sas_token = AZURE_SAS_TOKEN.lstrip('?')  # Remove leading ? if present

try:
    blob_service_client = BlobServiceClient(
        account_url=account_url,
        credential=sas_token
    )
    # Test connection on startup
    blob_service_client.get_container_client(CONTAINER_NAME).exists()
    logger.info(f"Successfully connected to Azure Blob Storage: {CONTAINER_NAME}")
except Exception as e:
    logger.error(f"Failed to connect to Azure Blob Storage: {str(e)}")
    raise RuntimeError(f"Azure Blob Storage connection failed: {str(e)}")


# ========== VALIDATION ==========
def validate_filename(filename: str) -> None:
    """Check for path traversal and length issues"""
    if not filename:
        raise HTTPException(400, "Filename cannot be empty")
    if ".." in filename or filename.startswith("/") or filename.startswith("\\"):
        raise HTTPException(400, "Invalid filename: path traversal detected")
    if len(filename) > 255:
        raise HTTPException(400, "Filename too long (max 255 characters)")


def get_file_extension(filename: str) -> str:
    """Extract and validate file extension"""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    return ext


# ========== BLOB OPERATIONS ==========
def generate_blob_name(user_id: str, original_filename: str) -> str:
    """Create unique blob name: {user_id}/{uuid}_{filename}{ext}"""
    unique_id = str(uuid.uuid4())
    ext = get_file_extension(original_filename)

    # Sanitize filename
    safe_name = "".join(c for c in original_filename if c.isalnum() or c in (' ', '-', '_', '.'))
    safe_name = safe_name.replace(" ", "_")[:50]
    safe_name = os.path.splitext(safe_name)[0]
    
    if not safe_name:
        safe_name = "file"

    return f"{user_id}/{unique_id}_{safe_name}{ext}"


def verify_ownership(blob_name: str, user_id: str) -> None:
    """Ensure user owns this file"""
    if not blob_name.startswith(f"{user_id}/"):
        logger.warning(f"Access denied: user {user_id} tried to access {blob_name}")
        raise HTTPException(403, "Access denied")


def get_container_client():
    """Get container client with error handling"""
    try:
        return blob_service_client.get_container_client(CONTAINER_NAME)
    except Exception as e:
        logger.error(f"Failed to get container client: {str(e)}")
        raise HTTPException(500, "Storage service temporarily unavailable")


# ========== FILE UPLOAD ==========
async def save_uploaded_file(file: UploadFile, user_id: str) -> dict:
    """Upload and validate file"""
    if not file.filename:
        raise HTTPException(400, "No filename provided")
    
    validate_filename(file.filename)
    ext = get_file_extension(file.filename)

    # Read file with size check (streaming)
    content = b""
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            content += chunk
            if len(content) > MAX_FILE_SIZE:
                raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File read error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Failed to read uploaded file")

    if len(content) == 0:
        raise HTTPException(400, "File is empty")

    # Validate file structure
    try:
        df = _load_dataframe_from_bytes(content, ext)
        if len(df) == 0:
            raise HTTPException(400, "File contains no data rows")
        if len(df.columns) == 0:
            raise HTTPException(400, "File contains no columns")
        if len(df) > 1_000_000:
            raise HTTPException(400, "File exceeds 1 million row limit")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File validation error for user {user_id}: {str(e)}")
        raise HTTPException(400, "Invalid file format or corrupted file")

    # Upload to Azure Blob Storage
    blob_name = generate_blob_name(user_id, file.filename)
    blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

    try:
        metadata = {
            "original_filename": file.filename,
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
            "row_count": str(len(df)),
            "column_count": str(len(df.columns))
        }
        blob_client.upload_blob(content, overwrite=False, metadata=metadata)
        logger.info(f"File uploaded successfully: {blob_name} for user {user_id}")
    except ResourceExistsError:
        raise HTTPException(409, "File with this name already exists")
    except AzureError as e:
        logger.error(f"Azure upload error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Upload failed. Please try again")
    except Exception as e:
        logger.error(f"Unexpected upload error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Upload failed. Please try again")

    return {
        "blob_name": blob_name,
        "original_filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "size_bytes": len(content),
        "size_mb": round(len(content) / (1024 * 1024), 2)
    }


# ========== FILE OPERATIONS ==========
def get_user_files(user_id: str) -> List[Dict]:
    """List all files for user"""
    container_client = get_container_client()
    prefix = f"{user_id}/"
    files = []

    try:
        for blob in container_client.list_blobs(name_starts_with=prefix):
            original_filename = None
            if blob.metadata:
                original_filename = blob.metadata.get('original_filename')
            
            if not original_filename:
                original_filename = _extract_filename(blob.name)

            files.append({
                "blob_name": blob.name,
                "original_filename": original_filename,
                "size_bytes": blob.size,
                "size_mb": round(blob.size / (1024 * 1024), 2),
                "uploaded_at": blob.creation_time.isoformat() if blob.creation_time else None,
            })

        files.sort(key=lambda x: x['uploaded_at'] or '', reverse=True)
        logger.info(f"Listed {len(files)} files for user {user_id}")
    except AzureError as e:
        logger.error(f"Azure list error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Failed to retrieve file list")
    except Exception as e:
        logger.error(f"Unexpected list error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Failed to retrieve file list")

    return files


def delete_user_file(blob_name: str, user_id: str) -> dict:
    """Delete a file"""
    verify_ownership(blob_name, user_id)
    blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

    try:
        if not blob_client.exists():
            raise HTTPException(404, "File not found")
        blob_client.delete_blob()
        logger.info(f"File deleted: {blob_name} by user {user_id}")
        return {"message": "File deleted successfully", "blob_name": blob_name}
    except HTTPException:
        raise
    except AzureError as e:
        logger.error(f"Azure delete error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Delete failed. Please try again")
    except Exception as e:
        logger.error(f"Unexpected delete error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Delete failed. Please try again")


def get_blob_name_by_filename(original_filename: str, user_id: str) -> Optional[str]:
    """Find blob_name by original filename (returns None if not found)"""
    files = get_user_files(user_id)
    matching = [f for f in files if f['original_filename'] == original_filename]
    
    if not matching:
        return None
    
    if len(matching) > 1:
        logger.warning(f"Multiple files found with name '{original_filename}' for user {user_id}")
        # Return the most recent one
        return matching[0]['blob_name']
    
    return matching[0]['blob_name']


def get_file_path(filename: str, user_id: str) -> str:
    """
    Get blob_name from either blob_name or original_filename.
    
    SIMPLIFIED: Only accepts blob_name for precision.
    If you only have the original filename, call get_blob_name_by_filename() first.
    """
    # If it looks like a blob_name (has user_id prefix), verify and return
    if '/' in filename and filename.startswith(f"{user_id}/"):
        verify_ownership(filename, user_id)
        
        # Verify it actually exists
        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, filename)
        try:
            if not blob_client.exists():
                raise HTTPException(404, "File not found")
        except AzureError as e:
            logger.error(f"Azure exists check error: {str(e)}")
            raise HTTPException(500, "Failed to verify file existence")
        
        return filename
    
    # Otherwise treat it as original filename and search
    blob_name = get_blob_name_by_filename(filename, user_id)
    if not blob_name:
        raise HTTPException(404, f"File not found: {filename}")
    
    return blob_name


# ========== DATA LOADING ==========
def _load_dataframe_from_bytes(content: bytes, extension: str) -> pd.DataFrame:
    """Load dataframe from bytes"""
    buffer = io.BytesIO(content)

    try:
        if extension == '.csv':
            # Try UTF-8 first, fallback to latin-1
            try:
                return pd.read_csv(buffer, encoding='utf-8')
            except UnicodeDecodeError:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding='latin-1')
        elif extension in ['.xlsx', '.xls']:
            engine = 'openpyxl' if extension == '.xlsx' else 'xlrd'
            return pd.read_excel(buffer, engine=engine)
        else:
            raise ValueError(f"Unsupported extension: {extension}")
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")


def load_dataframe(blob_name: str, user_id: str) -> pd.DataFrame:
    """Download and load dataframe from Azure Blob Storage"""
    verify_ownership(blob_name, user_id)
    blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

    try:
        download_stream = blob_client.download_blob()
        content = download_stream.readall()
        ext = get_file_extension(blob_name)
        df = _load_dataframe_from_bytes(content, ext)
        logger.info(f"Loaded dataframe: {blob_name} ({len(df)} rows) for user {user_id}")
        return df
    except ResourceNotFoundError:
        raise HTTPException(404, "File not found")
    except HTTPException:
        raise
    except AzureError as e:
        logger.error(f"Azure download error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Failed to load file")
    except Exception as e:
        logger.error(f"Unexpected load error for user {user_id}: {str(e)}")
        raise HTTPException(500, "Failed to load file")


def get_dataframe_preview(blob_name: str, user_id: str, num_rows: int = 10) -> dict:
    """Get file preview with metadata"""
    if num_rows < 1 or num_rows > 100:
        raise HTTPException(400, "num_rows must be between 1 and 100")
    
    df = load_dataframe(blob_name, user_id)

    return {
        "blob_name": blob_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview_rows": df.head(num_rows).to_dict('records'),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }


def _extract_filename(blob_name: str) -> str:
    """Extract original filename from blob_name (fallback)"""
    try:
        filename_part = blob_name.split('/')[-1]
        # Format is: {uuid}_{filename}{ext}
        parts = filename_part.split('_', 1)
        return parts[1] if len(parts) > 1 else filename_part
    except Exception:
        return blob_name
