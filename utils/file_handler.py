import uuid
import io
import os
import logging
from datetime import datetime
from typing import List, Dict
from fastapi import UploadFile, HTTPException
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError

logger = logging.getLogger(__name__)

# ========== CONFIG ==========
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = "salesdata"
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Build account URL from account name and SAS token
if not AZURE_ACCOUNT_NAME or not AZURE_SAS_TOKEN:
    raise ValueError("Missing AZURE_ACCOUNT_NAME or AZURE_SAS_TOKEN in environment variables")

account_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net/?{AZURE_SAS_TOKEN}"
logger.info(f"Initializing Azure Blob Storage: {AZURE_ACCOUNT_NAME}")

try:
    blob_service_client = BlobServiceClient(account_url=account_url)
except Exception as e:
    logger.error(f"Failed to initialize BlobServiceClient: {str(e)}")
    raise


# ========== VALIDATION ==========
def validate_filename(filename: str) -> None:
    """Check for path traversal and length issues"""
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(400, "Invalid filename")
    if len(filename) > 255:
        raise HTTPException(400, "Filename too long")


def get_file_extension(filename: str) -> str:
    """Extract and validate file extension"""
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only {', '.join(ALLOWED_EXTENSIONS)} files allowed")
    return ext


# ========== BLOB OPERATIONS ==========
def generate_blob_name(user_id: str, original_filename: str) -> str:
    """Create unique blob name: {user_id}/{uuid}_{filename}{ext}"""
    unique_id = str(uuid.uuid4())
    ext = get_file_extension(original_filename)

    safe_name = "".join(c for c in original_filename if c.isalnum() or c in (' ', '-', '_', '.'))
    safe_name = safe_name.replace(" ", "_")[:50]
    safe_name = os.path.splitext(safe_name)[0]

    return f"{user_id}/{unique_id}_{safe_name}{ext}"


def verify_ownership(blob_name: str, user_id: str) -> None:
    """Ensure user owns this file"""
    if not blob_name.startswith(f"{user_id}/"):
        raise HTTPException(403, "Access denied: You don't own this file")


def get_container_client():
    """Get container client"""
    return blob_service_client.get_container_client(CONTAINER_NAME)


# ========== FILE UPLOAD ==========
async def save_uploaded_file(file: UploadFile, user_id: str) -> dict:
    """Upload and validate file"""
    validate_filename(file.filename)
    ext = get_file_extension(file.filename)

    # Read file with size check
    content = b""
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        content += chunk
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)}MB limit")

    if len(content) == 0:
        raise HTTPException(400, "File is empty")

    # Validate structure
    try:
        df = _load_dataframe_from_bytes(content, ext)
        if len(df) == 0:
            raise HTTPException(400, "File has no data rows")
        if len(df.columns) == 0:
            raise HTTPException(400, "File has no columns")
        if len(df) > 1_000_000:
            raise HTTPException(400, "File exceeds 1M row limit")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid file format: {str(e)}")

    # Upload to Azure
    blob_name = generate_blob_name(user_id, file.filename)
    blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

    try:
        metadata = {
            "original_filename": file.filename,
            "user_id": user_id,
            "uploaded_at": datetime.utcnow().isoformat(),
        }
        blob_client.upload_blob(content, overwrite=False, metadata=metadata)
        logger.info(f"File uploaded: {blob_name} by user {user_id}")
    except ResourceExistsError:
        raise HTTPException(409, "File already exists")
    except Exception as e:
        logger.error(f"Upload failed for {user_id}: {str(e)}")
        raise HTTPException(500, f"Upload failed: {str(e)}")

    return {
        "blob_name": blob_name,
        "original_filename": file.filename,
        "rows": len(df),
        "columns": list(df.columns),
        "size_bytes": len(content),
    }


# ========== FILE OPERATIONS ==========
def get_user_files(user_id: str) -> List[Dict]:
    """List all files for user"""
    try:
        container_client = get_container_client()
        prefix = f"{user_id}/"
        files = []

        for blob in container_client.list_blobs(name_starts_with=prefix):
            original_filename = blob.metadata.get('original_filename') if blob.metadata else None
            if not original_filename:
                original_filename = _extract_filename(blob.name)

            files.append({
                "blob_name": blob.name,
                "original_filename": original_filename,
                "size_mb": round(blob.size / (1024 * 1024), 2),
                "uploaded_at": blob.creation_time.isoformat() if blob.creation_time else None,
            })

        files.sort(key=lambda x: x['uploaded_at'] or '', reverse=True)
        logger.info(f"Listed {len(files)} files for user {user_id}")
        return files
    except Exception as e:
        logger.error(f"Failed to list files for {user_id}: {str(e)}")
        raise HTTPException(500, f"Failed to list files: {str(e)}")


def delete_user_file(blob_name: str, user_id: str) -> dict:
    """Delete a file"""
    try:
        verify_ownership(blob_name, user_id)

        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

        if not blob_client.exists():
            raise HTTPException(404, "File not found")

        blob_client.delete_blob()
        logger.info(f"File deleted: {blob_name} by user {user_id}")
        return {"message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed for {user_id}: {str(e)}")
        raise HTTPException(500, f"Delete failed: {str(e)}")


def get_file_path(blob_name: str, user_id: str) -> str:
    """Get blob name for analytics (handles both blob_name and original_filename)"""
    # If full blob_name provided, verify and return
    if blob_name.startswith(f"{user_id}/"):
        verify_ownership(blob_name, user_id)
        return blob_name

    # Otherwise search by original filename
    files = get_user_files(user_id)
    matching = [f for f in files if f['original_filename'] == blob_name]
    if not matching:
        raise HTTPException(404, "File not found")

    return matching[0]['blob_name']


# ========== DATA LOADING ==========
def _load_dataframe_from_bytes(content: bytes, extension: str) -> pd.DataFrame:
    """Load dataframe from bytes"""
    buffer = io.BytesIO(content)

    try:
        if extension == '.csv':
            try:
                return pd.read_csv(buffer, encoding='utf-8')
            except UnicodeDecodeError:
                buffer.seek(0)
                return pd.read_csv(buffer, encoding='latin-1')
        else:
            return pd.read_excel(buffer, engine='openpyxl' if extension == '.xlsx' else None)
    except Exception as e:
        raise ValueError(f"Failed to parse file: {str(e)}")


def load_dataframe(blob_name: str, user_id: str) -> pd.DataFrame:
    """Download and load dataframe from Azure"""
    try:
        verify_ownership(blob_name, user_id)

        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

        content = blob_client.download_blob().readall()
        ext = get_file_extension(blob_name)
        logger.info(f"Loaded dataframe: {blob_name} for user {user_id}")
        return _load_dataframe_from_bytes(content, ext)
    except ResourceNotFoundError:
        raise HTTPException(404, "File not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load file {blob_name} for {user_id}: {str(e)}")
        raise HTTPException(500, f"Failed to load file: {str(e)}")


def get_dataframe_preview(blob_name: str, user_id: str, num_rows: int = 10) -> dict:
    """Get file preview"""
    df = load_dataframe(blob_name, user_id)

    return {
        "blob_name": blob_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview_rows": df.head(num_rows).to_dict('records'),
    }


def _extract_filename(blob_name: str) -> str:
    """Extract original filename from blob_name"""
    try:
        filename_part = blob_name.split('/')[-1]
        parts = filename_part.split('_', 1)
        return parts[1] if len(parts) > 1 else filename_part
    except:
        return blob_name