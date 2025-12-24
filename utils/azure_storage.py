import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
from io import BytesIO
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError, AzureError
from utils.auth import hash_string
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Azure Blob Storage configuration
AZURE_STORAGE_URL = os.getenv("AZURE_STORAGE_URL", "")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN", "")
CONTAINER_NAME = "salesdata"

# Parse storage account name safely
if AZURE_STORAGE_URL:
    # Remove container name if present in URL
    if "/salesdata" in AZURE_STORAGE_URL:
        AZURE_STORAGE_URL = AZURE_STORAGE_URL.split("/salesdata")[0]
        logger.warning(f"Removed container name from AZURE_STORAGE_URL. Using: {AZURE_STORAGE_URL}")
    
    parsed = urlparse(AZURE_STORAGE_URL)
    STORAGE_ACCOUNT_NAME = parsed.hostname.split('.')[0] if parsed.hostname else None
    ACCOUNT_URL = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net" if STORAGE_ACCOUNT_NAME else None
else:
    STORAGE_ACCOUNT_NAME = None
    ACCOUNT_URL = None
    logger.warning("AZURE_STORAGE_URL not configured")


def get_blob_service_client() -> BlobServiceClient:
    """Create Azure Blob Service Client with proper error handling"""
    if not ACCOUNT_URL:
        raise ValueError("AZURE_STORAGE_URL not properly configured")
    if not AZURE_SAS_TOKEN:
        raise ValueError("AZURE_SAS_TOKEN not configured")
    
    try:
        return BlobServiceClient(account_url=ACCOUNT_URL, credential=AZURE_SAS_TOKEN)
    except Exception as e:
        logger.error(f"Failed to create BlobServiceClient: {e}")
        raise


def get_container_client():
    """Get container client with validation"""
    skip_validation = os.getenv("SKIP_AZURE_VALIDATION", "false").lower() == "true"
    
    try:
        blob_service_client = get_blob_service_client()
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        
        if not skip_validation:
            # Validate container exists
            try:
                container_exists = container_client.exists()
                if not container_exists:
                    logger.warning(f"Container '{CONTAINER_NAME}' does not exist, attempting to create...")
                    try:
                        container_client.create_container()
                        logger.info(f"✓ Created container '{CONTAINER_NAME}'")
                    except Exception as create_err:
                        logger.error(f"Failed to create container: {create_err}")
                        raise RuntimeError(f"Container '{CONTAINER_NAME}' does not exist and cannot be created: {create_err}")
                else:
                    logger.info(f"✓ Container '{CONTAINER_NAME}' exists and is accessible")
            except Exception as e:
                logger.error(f"Container validation failed: {e}")
                raise RuntimeError(f"Azure Blob Storage connection failed: {e}")
        else:
            logger.warning("⚠ Skipping Azure container validation (SKIP_AZURE_VALIDATION=true)")
        
        return container_client
    except Exception as e:
        logger.error(f"Failed to get container client: {e}")
        raise


def health_check() -> dict:
    """Check Azure Blob Storage connectivity and configuration"""
    health = {
        "account": STORAGE_ACCOUNT_NAME,
        "container": CONTAINER_NAME,
        "account_url": ACCOUNT_URL
    }
    
    # Check if credentials are configured
    if not ACCOUNT_URL or not AZURE_SAS_TOKEN:
        return {
            **health,
            "status": "unhealthy",
            "storage": "disconnected",
            "error": "Azure credentials not configured",
            "missing": [
                var for var in ["AZURE_STORAGE_URL", "AZURE_SAS_TOKEN"] 
                if not os.getenv(var)
            ]
        }
    
    # Check if we're skipping validation
    if os.getenv("SKIP_AZURE_VALIDATION", "false").lower() == "true":
        return {
            **health,
            "status": "unknown",
            "storage": "not validated",
            "message": "Azure validation skipped"
        }
    
    # Try to connect
    try:
        container_client = get_container_client()
        exists = container_client.exists()
        
        return {
            **health,
            "status": "healthy",
            "storage": "connected",
            "container_exists": exists
        }
    except Exception as e:
        return {
            **health,
            "status": "unhealthy",
            "storage": "disconnected",
            "error": str(e)
        }


async def save_uploaded_file(file: UploadFile, user_id: str) -> dict:
    """Save uploaded file to Azure Blob Storage with user-specific filename"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "Only CSV/Excel files allowed")

    # Generate unique filename
    user_hash = hash_string(user_id)[:12]
    timestamp = int(datetime.utcnow().timestamp())
    safe_filename = file.filename.replace(" ", "_")
    blob_name = f"{user_hash}_{timestamp}_{safe_filename}"

    try:
        # Read file content
        content = await file.read()
        
        # Validate file by reading it first
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))
        
        # Upload to Azure Blob Storage
        container_client = get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        
        # Upload with metadata
        blob_client.upload_blob(
            content,
            overwrite=True,
            metadata={
                "user_id": user_id,
                "original_filename": file.filename,
                "uploaded_at": datetime.utcnow().isoformat()
            }
        )

        return {
            "filename": blob_name,
            "original_filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": "File uploaded successfully to Azure Blob Storage"
        }
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"Invalid file format: {str(e)}")
    except Exception as e:
        # Clean up blob if upload succeeded but something else failed
        try:
            blob_client = container_client.get_blob_client(blob_name)
            if blob_client.exists():
                blob_client.delete_blob()
        except:
            pass
        raise HTTPException(400, f"File upload failed: {str(e)}")


def get_user_files(user_id: str) -> List[Dict]:
    """List all files for a specific user"""
    user_hash = hash_string(user_id)[:12]
    files = []
    
    try:
        container_client = get_container_client()
        blobs = container_client.list_blobs(name_starts_with=user_hash)
        
        for blob in blobs:
            try:
                # Get metadata if available
                blob_client = container_client.get_blob_client(blob.name)
                props = blob_client.get_blob_properties()
                metadata = props.metadata or {}
                
                files.append({
                    "filename": blob.name,
                    "original_filename": metadata.get('original_filename', blob.name),
                    "size": blob.size,
                    "uploaded_at": metadata.get('uploaded_at', props.last_modified.isoformat()),
                    "modified": props.last_modified.isoformat()
                })
            except Exception as e:
                logger.warning(f"Failed to get properties for blob {blob.name}: {e}")
                # Still add the file with basic info
                files.append({
                    "filename": blob.name,
                    "original_filename": blob.name,
                    "size": blob.size,
                    "modified": blob.last_modified.isoformat()
                })
        
        return files
    except Exception as e:
        logger.error(f"Failed to list files for user {user_id}: {e}")
        raise HTTPException(500, f"Failed to list files: {str(e)}")


def delete_user_file(filename: str, user_id: str) -> dict:
    """Delete a specific file from Azure Blob Storage"""
    user_hash = hash_string(user_id)[:12]

    if not filename.startswith(user_hash):
        raise HTTPException(403, "Access denied to this file")

    try:
        container_client = get_container_client()
        blob_client = container_client.get_blob_client(filename)
        
        # Check if blob exists
        if not blob_client.exists():
            raise HTTPException(404, "File not found")
        
        blob_client.delete_blob()
        return {"message": "File deleted successfully", "filename": filename}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


def get_file_path(filename: str, user_id: str) -> str:
    """
    Validate file exists and user has access, return blob name.
    Handles both blob names and original filenames.
    """
    user_hash = hash_string(user_id)[:12]
    
    # If filename already has the user hash prefix, verify it directly
    if filename.startswith(user_hash):
        try:
            container_client = get_container_client()
            blob_client = container_client.get_blob_client(filename)
            
            if not blob_client.exists():
                raise HTTPException(404, f"File '{filename}' not found")
            
            return filename
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Error accessing file: {str(e)}")
    
    # Otherwise, search for file by original filename
    try:
        files = get_user_files(user_id)
        
        for file in files:
            if file.get('original_filename') == filename or file.get('filename') == filename:
                return file['filename']
        
        # If not found, raise 404
        raise HTTPException(404, f"File '{filename}' not found for this user")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error searching for file: {str(e)}")


def download_blob_to_bytes(blob_name: str) -> bytes:
    """Download a blob from Azure Storage and return as bytes"""
    try:
        container_client = get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        
        if not blob_client.exists():
            raise HTTPException(404, f"File '{blob_name}' not found")
        
        return blob_client.download_blob().readall()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


def load_dataframe(filename: str, user_id: str = None) -> pd.DataFrame:
    """
    Load dataframe from Azure Blob Storage.
    
    Args:
        filename: Blob name or original filename
        user_id: User ID for access validation (optional)
    
    Returns:
        pandas DataFrame
    """
    try:
        # Validate user access if user_id provided
        if user_id:
            user_hash = hash_string(user_id)[:12]
            
            # If filename doesn't have user prefix, find the actual blob name
            if not filename.startswith(user_hash):
                filename = get_file_path(filename, user_id)
            else:
                # Verify access even if it has the prefix
                if not filename.startswith(user_hash):
                    raise HTTPException(403, "Access denied to this file")
        
        # Download blob content
        content = download_blob_to_bytes(filename)
        
        # Load based on extension
        if filename.endswith('.csv'):
            return pd.read_csv(BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(BytesIO(content))
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty")
    except pd.errors.ParserError as e:
        raise HTTPException(400, f"Invalid file format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to load dataframe from {filename}: {e}")
        raise HTTPException(500, f"Failed to load file: {str(e)}")


# Initialize and validate on module load
try:
    skip_validation = os.getenv("SKIP_AZURE_VALIDATION", "false").lower() == "true"
    
    if not skip_validation:
        try:
            container_client = get_container_client()
            container_exists = container_client.exists()
            if not container_exists:
                logger.warning(f"Container '{CONTAINER_NAME}' does not exist, attempting to create...")
                try:
                    container_client.create_container()
                    logger.info(f"✓ Created container '{CONTAINER_NAME}'")
                except Exception as e:
                    logger.error(f"Failed to create container: {e}")
                    raise RuntimeError(f"Container '{CONTAINER_NAME}' does not exist and cannot be created: {e}")
            else:
                logger.info(f"✓ Azure Blob Storage initialized - Container '{CONTAINER_NAME}' is accessible")
        except Exception as e:
            logger.error(f"Container validation failed: {e}")
            logger.error(f"Failed to initialize Azure Blob Storage: {e}")
            raise RuntimeError(f"Azure Blob Storage connection failed: {e}")
    else:
        logger.warning("⚠ Continuing despite Azure validation being skipped (SKIP_AZURE_VALIDATION=true)")
        
except Exception as e:
    logger.error(f"Failed to initialize Azure Blob Storage: {e}")
    if os.getenv("SKIP_AZURE_VALIDATION", "false").lower() != "true":
        raise RuntimeError(f"Azure Blob Storage connection failed: {e}")
    else:
        logger.warning("⚠ Continuing despite Azure initialization failure (SKIP_AZURE_VALIDATION=true)")
