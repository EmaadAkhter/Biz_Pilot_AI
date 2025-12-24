import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
from io import BytesIO
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from utils.auth import hash_string
from dotenv import load_dotenv

load_dotenv()

# Azure Blob Storage configuration
AZURE_STORAGE_URL = os.getenv("AZURE_STORAGE_URL")
AZURE_SAS_TOKEN = os.getenv("AZURE_SAS_TOKEN")
CONTAINER_NAME = "salesdata"

# Parse the storage account name from URL
STORAGE_ACCOUNT_NAME = AZURE_STORAGE_URL.split("//")[1].split(".")[0]
ACCOUNT_URL = f"https://{STORAGE_ACCOUNT_NAME}.blob.core.windows.net"


def get_blob_service_client() -> BlobServiceClient:
    """Create and return Azure Blob Service Client with SAS token"""
    return BlobServiceClient(account_url=ACCOUNT_URL, credential=AZURE_SAS_TOKEN)


def get_container_client() -> ContainerClient:
    """Get container client for salesdata container"""
    blob_service_client = get_blob_service_client()
    return blob_service_client.get_container_client(CONTAINER_NAME)


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

        # Validate file by reading it
        if file.filename.endswith('.csv'):
            df = pd.read_csv(BytesIO(content))
        else:
            df = pd.read_excel(BytesIO(content))

        return {
            "filename": blob_name,
            "original_filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": "File uploaded successfully to Azure Blob Storage"
        }
    except Exception as e:
        # Clean up blob if validation fails
        try:
            blob_client.delete_blob()
        except:
            pass
        raise HTTPException(400, f"Invalid file: {str(e)}")


def list_files(user_hash: Optional[str] = None) -> List[Dict]:
    """List all files in Azure Blob Storage, optionally filtered by user_hash"""
    files = []
    
    try:
        container_client = get_container_client()
        blobs = container_client.list_blobs()
        
        for blob in blobs:
            # Filter by user_hash if provided
            if user_hash is None or blob.name.startswith(user_hash):
                files.append({
                    "filename": blob.name,
                    "size": blob.size,
                    "modified": blob.last_modified.timestamp(),
                    "uploaded_at": blob.last_modified.isoformat()
                })
        
        return files
    except Exception as e:
        raise HTTPException(500, f"Failed to list files: {str(e)}")


def get_user_files(user_id: str) -> List[Dict]:
    """List all files for a specific user"""
    user_hash = hash_string(user_id)[:12]
    return list_files(user_hash)


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


def get_file_path(filename: str, user_id: str = None) -> str:
    """
    For Azure Blob Storage, this returns the blob name after verification.
    The actual file will be downloaded when needed.
    """
    if user_id:
        user_hash = hash_string(user_id)[:12]
        if not filename.startswith(user_hash):
            raise HTTPException(403, "Access denied to this file")

    try:
        container_client = get_container_client()
        blob_client = container_client.get_blob_client(filename)
        
        if not blob_client.exists():
            raise HTTPException(404, "File not found")
        
        return filename  # Return blob name for later use
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Error accessing file: {str(e)}")


def download_blob_to_bytes(blob_name: str) -> bytes:
    """Download a blob from Azure Storage and return as bytes"""
    try:
        container_client = get_container_client()
        blob_client = container_client.get_blob_client(blob_name)
        
        if not blob_client.exists():
            raise HTTPException(404, "File not found")
        
        return blob_client.download_blob().readall()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Download failed: {str(e)}")


def load_dataframe(filename: str) -> pd.DataFrame:
    """Load dataframe from Azure Blob Storage"""
    try:
        # Download blob content
        content = download_blob_to_bytes(filename)
        
        # Load into pandas based on file extension
        if filename.endswith('.csv'):
            return pd.read_csv(BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(BytesIO(content))
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load dataframe: {str(e)}")
