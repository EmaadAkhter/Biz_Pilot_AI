import uuid
import io
import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
from utils.redis_cache import(
    get_cached_file_list,
    cache_file_list,
    invalidate_file_list,
    invalidate_analytics,
    invalidate_forecast,
    cache
)

logger = logging.getLogger(__name__)

# ========== CONFIG ==========
AZURE_ACCOUNT_NAME = os.getenv("AZURE_ACCOUNT_NAME")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = "salesdata"
MAX_FILE_SIZE = 100 * 1024 * 1024
ALLOWED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}

# Build account URL - DO NOT include container name here
if not AZURE_ACCOUNT_NAME or not AZURE_SAS_TOKEN:
    raise ValueError("Missing AZURE_ACCOUNT_NAME or AZURE_SAS_TOKEN in environment variables")

account_url = f"https://{AZURE_ACCOUNT_NAME}.blob.core.windows.net"
logger.info(f"Initializing Azure Blob Storage: {AZURE_ACCOUNT_NAME}")

try:
    blob_service_client = BlobServiceClient(account_url=account_url, credential=AZURE_SAS_TOKEN)
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
    """Upload and validate file with cache invalidation"""
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
            "rows": str(len(df)),
            "columns": str(len(df.columns))
        }
        blob_client.upload_blob(content, overwrite=False, metadata=metadata)
        logger.info(f"✓ File uploaded: {blob_name} by user {user_id}")

        # Invalidate file list cache (new file added)
        invalidate_file_list(user_id)
        logger.info(f"✓ File list cache invalidated for user {user_id}")

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
        "size_mb": round(len(content) / (1024 * 1024), 2)
    }


# ========== FILE OPERATIONS ==========
def get_user_files(user_id: str, use_cache: bool = True) -> List[Dict]:
    """List all files for user with Redis caching

    Args:
        user_id: User identifier
        use_cache: Whether to use Redis cache (default: True)

    Returns:
        List of file metadata dictionaries
    """

    # Try cache first
    if use_cache:
        cached = get_cached_file_list(user_id)
        if cached:
            logger.info(f"✓ File list cache HIT for user {user_id}")
            return cached
        logger.info(f"○ File list cache MISS for user {user_id}")

    # Fetch from Azure Blob Storage
    logger.info(f"Fetching file list from Azure for user {user_id}")

    try:
        container_client = get_container_client()
        prefix = f"{user_id}/"
        files = []

        for blob in container_client.list_blobs(name_starts_with=prefix):
            # Get metadata
            original_filename = blob.metadata.get('original_filename') if blob.metadata else None
            if not original_filename:
                original_filename = _extract_filename(blob.name)

            files.append({
                "blob_name": blob.name,
                "original_filename": original_filename,
                "size_bytes": blob.size,
                "size_mb": round(blob.size / (1024 * 1024), 2),
                "uploaded_at": blob.creation_time.isoformat() if blob.creation_time else None,
                "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
            })

        # Sort by upload time (newest first)
        files.sort(key=lambda x: x['uploaded_at'] or '', reverse=True)
        logger.info(f"✓ Listed {len(files)} files for user {user_id}")

        # Cache the result
        if use_cache and files:
            if cache_file_list(user_id, files):
                logger.info(f"✓ File list cached for user {user_id}")
            else:
                logger.warning(f"⚠ Failed to cache file list for user {user_id}")

        return files

    except Exception as e:
        logger.error(f"Failed to list files for {user_id}: {str(e)}")
        raise HTTPException(500, f"Failed to list files: {str(e)}")


def delete_user_file(blob_name: str, user_id: str) -> dict:
    """Delete a file and invalidate all related caches

    Args:
        blob_name: Blob identifier
        user_id: User identifier

    Returns:
        Success message
    """
    try:
        verify_ownership(blob_name, user_id)

        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

        if not blob_client.exists():
            raise HTTPException(404, "File not found")

        # Delete from Azure
        blob_client.delete_blob()
        logger.info(f"✓ File deleted: {blob_name} by user {user_id}")

        # Invalidate all related caches
        invalidate_file_list(user_id)
        invalidate_analytics(user_id, blob_name)
        invalidate_forecast(user_id, blob_name)
        logger.info(f"✓ All caches invalidated for {blob_name}")

        return {
            "message": "File deleted successfully",
            "blob_name": blob_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete failed for {user_id}: {str(e)}")
        raise HTTPException(500, f"Delete failed: {str(e)}")


def get_file_path(blob_name: str, user_id: str) -> str:
    """Get blob name for analytics (handles both blob_name and original_filename)

    Args:
        blob_name: Either full blob_name or original filename
        user_id: User identifier

    Returns:
        Full blob_name path
    """
    # If full blob_name provided, verify and return
    if blob_name.startswith(f"{user_id}/"):
        verify_ownership(blob_name, user_id)
        return blob_name

    # Otherwise search by original filename
    files = get_user_files(user_id)
    matching = [f for f in files if f['original_filename'] == blob_name]

    if not matching:
        raise HTTPException(404, f"File not found: {blob_name}")

    return matching[0]['blob_name']


def get_file_metadata(blob_name: str, user_id: str, use_cache: bool = True) -> Dict:
    """Get detailed file metadata

    Args:
        blob_name: Blob identifier
        user_id: User identifier
        use_cache: Whether to use cache

    Returns:
        File metadata dictionary
    """
    verify_ownership(blob_name, user_id)

    # Check if metadata is in file list cache
    if use_cache:
        cached_files = get_cached_file_list(user_id)
        if cached_files:
            matching = [f for f in cached_files if f['blob_name'] == blob_name]
            if matching:
                logger.info(f"✓ File metadata from cache: {blob_name}")
                return matching[0]

    # Fetch from Azure
    try:
        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)
        props = blob_client.get_blob_properties()

        metadata = {
            "blob_name": blob_name,
            "original_filename": props.metadata.get('original_filename', _extract_filename(blob_name)),
            "size_bytes": props.size,
            "size_mb": round(props.size / (1024 * 1024), 2),
            "uploaded_at": props.creation_time.isoformat() if props.creation_time else None,
            "last_modified": props.last_modified.isoformat() if props.last_modified else None,
            "content_type": props.content_settings.content_type if props.content_settings else None,
        }

        logger.info(f"✓ File metadata fetched: {blob_name}")
        return metadata

    except ResourceNotFoundError:
        raise HTTPException(404, "File not found")
    except Exception as e:
        logger.error(f"Failed to get metadata for {blob_name}: {str(e)}")
        raise HTTPException(500, f"Failed to get file metadata: {str(e)}")


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
    """Download and load dataframe from Azure

    Args:
        blob_name: Blob identifier
        user_id: User identifier

    Returns:
        pandas DataFrame
    """
    try:
        verify_ownership(blob_name, user_id)

        blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, blob_name)

        # Download file content
        content = blob_client.download_blob().readall()
        ext = get_file_extension(blob_name)

        # Load into DataFrame
        df = _load_dataframe_from_bytes(content, ext)
        logger.info(f"✓ Loaded dataframe: {blob_name} ({len(df)} rows, {len(df.columns)} cols)")

        return df

    except ResourceNotFoundError:
        raise HTTPException(404, "File not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load file {blob_name} for {user_id}: {str(e)}")
        raise HTTPException(500, f"Failed to load file: {str(e)}")


def get_dataframe_preview(blob_name: str, user_id: str, num_rows: int = 10,
                          use_cache: bool = True) -> dict:
    """Get file preview with optional caching

    Args:
        blob_name: Blob identifier
        user_id: User identifier
        num_rows: Number of preview rows
        use_cache: Whether to cache preview

    Returns:
        Dictionary with preview data
    """
    cache_key = f"preview:{user_id}:{blob_name}:{num_rows}"

    # Try cache first
    if use_cache:
        cached = cache.get(cache_key)
        if cached:
            logger.info(f"✓ Preview cache HIT: {blob_name}")
            return cached

    # Load and generate preview
    df = load_dataframe(blob_name, user_id)

    preview = {
        "blob_name": blob_name,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "preview_rows": df.head(num_rows).to_dict('records'),
        "sample_values": {
            col: df[col].dropna().head(3).tolist()
            for col in df.columns
        }
    }

    # Cache preview for 5 minutes
    if use_cache:
        if cache.set(cache_key, preview, ttl=300):
            logger.info(f"✓ Preview cached: {blob_name}")

    return preview


def _extract_filename(blob_name: str) -> str:
    """Extract original filename from blob_name

    Args:
        blob_name: Full blob path (user_id/uuid_filename.ext)

    Returns:
        Original filename
    """
    try:
        # Remove user_id prefix
        filename_part = blob_name.split('/')[-1]
        # Remove UUID prefix
        parts = filename_part.split('_', 1)
        return parts[1] if len(parts) > 1 else filename_part
    except:
        return blob_name


# ========== CACHE MANAGEMENT ==========
def clear_user_cache(user_id: str) -> dict:
    """Clear all cached data for a user

    Args:
        user_id: User identifier

    Returns:
        Summary of cleared items
    """
    cleared = {
        "file_list": 0,
        "analytics": 0,
        "forecasts": 0,
        "previews": 0
    }

    # Clear file list
    if invalidate_file_list(user_id):
        cleared["file_list"] = 1

    # Clear analytics (all files)
    cleared["analytics"] = cache.delete_pattern(f"analytics:{user_id}:*")

    # Clear forecasts (all files)
    cleared["forecasts"] = cache.delete_pattern(f"forecast:{user_id}:*")

    # Clear previews (all files)
    cleared["previews"] = cache.delete_pattern(f"preview:{user_id}:*")

    total = sum(cleared.values())
    logger.info(f"✓ Cleared {total} cache entries for user {user_id}")

    return {
        "cleared": cleared,
        "total": total,
        "user_id": user_id
    }


def get_cache_stats(user_id: Optional[str] = None) -> dict:
    """Get cache statistics

    Args:
        user_id: Optional user ID to get user-specific stats

    Returns:
        Cache statistics
    """
    stats = cache.get_stats()

    if user_id:
        # Count user-specific cached items
        patterns = {
            "files": f"files:{user_id}",
            "analytics": f"analytics:{user_id}:*",
            "forecasts": f"forecast:{user_id}:*",
            "previews": f"preview:{user_id}:*"
        }

        user_stats = {}
        for key, pattern in patterns.items():
            try:
                keys = cache.client.keys(pattern) if cache.enabled and cache.client else []
                user_stats[key] = len(keys)
            except:
                user_stats[key] = 0

        stats["user_cache"] = user_stats
        stats["user_id"] = user_id

    return stats
