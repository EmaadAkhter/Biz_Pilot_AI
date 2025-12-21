import os
from datetime import datetime
from typing import List, Dict, Optional
from fastapi import UploadFile, HTTPException
import pandas as pd
from utils.auth import hash_string

SALES_DATA_DIR = "user_sales_data"

# Ensure directory exists
os.makedirs(SALES_DATA_DIR, exist_ok=True)


async def save_uploaded_file(file: UploadFile, user_id: str) -> dict:
    """Save uploaded file with user-specific filename"""
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise HTTPException(400, "Only CSV/Excel files allowed")

    user_hash = hash_string(user_id)[:12]
    timestamp = int(datetime.utcnow().timestamp())
    safe_filename = file.filename.replace(" ", "_")
    new_filename = f"{user_hash}_{timestamp}_{safe_filename}"
    filepath = os.path.join(SALES_DATA_DIR, new_filename)

    content = await file.read()
    with open(filepath, "wb") as f:
        f.write(content)

    try:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        return {
            "filename": new_filename,
            "original_filename": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
            "message": "File uploaded successfully"
        }
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(400, f"Invalid file: {str(e)}")


def list_files(user_hash: Optional[str] = None) -> List[Dict]:
    """List all files, optionally filtered by user_hash"""
    files = []

    if not os.path.exists(SALES_DATA_DIR):
        return files

    for filename in os.listdir(SALES_DATA_DIR):
        if filename.endswith(('.csv', '.xlsx', '.xls')):
            if user_hash is None or filename.startswith(user_hash):
                filepath = os.path.join(SALES_DATA_DIR, filename)
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(filepath),
                    "modified": os.path.getmtime(filepath),
                    "uploaded_at": datetime.fromtimestamp(
                        os.path.getctime(filepath)
                    ).isoformat()
                })

    return files


def get_user_files(user_id: str) -> List[Dict]:
    """List all files for a specific user"""
    user_hash = hash_string(user_id)[:12]
    return list_files(user_hash)


def delete_user_file(filename: str, user_id: str) -> dict:
    """Delete a specific file"""
    user_hash = hash_string(user_id)[:12]

    if not filename.startswith(user_hash):
        raise HTTPException(403, "Access denied to this file")

    filepath = os.path.join(SALES_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")

    try:
        os.remove(filepath)
        return {"message": "File deleted successfully", "filename": filename}
    except Exception as e:
        raise HTTPException(500, f"Delete failed: {str(e)}")


def get_file_path(filename: str, user_id: str = None) -> str:
    """Get full filepath and optionally verify user access"""
    if user_id:
        user_hash = hash_string(user_id)[:12]
        if not filename.startswith(user_hash):
            raise HTTPException(403, "Access denied to this file")

    filepath = os.path.join(SALES_DATA_DIR, filename)

    if not os.path.exists(filepath):
        raise HTTPException(404, "File not found")

    return filepath


def load_dataframe(filename: str) -> pd.DataFrame:
    """Load dataframe from the sales data directory"""
    filepath = get_file_path(filename)

    if filename.endswith('.csv'):
        return pd.read_csv(filepath)
    elif filename.endswith(('.xlsx', '.xls')):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")