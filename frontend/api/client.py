from typing import Any

import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from frontend.state import ChatInfo

API_BASE_URL = "http://localhost:8000"


def fetch_chats() -> list[ChatInfo]:
    try:
        response = requests.get(f"{API_BASE_URL}/chats", timeout=10)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return []


def create_chat(
    model_file: UploadedFile,
    data_files: list[UploadedFile] | None,
    modeling_language: str,
    profile_label: str,
    solver: str,
    chat_name: str = "",
) -> ChatInfo | None:
    files = [
        (
            "model_file",
            (model_file.name, model_file.getvalue(), "text/x-python"),
        )
    ]
    for data_file in data_files or []:
        content_type = data_file.type or "application/octet-stream"
        files.append(
            (
                "data_files",
                (data_file.name, data_file.getvalue(), content_type),
            )
        )

    data: dict[str, Any] = {"profile_id": profile_label}
    if chat_name:
        data["name"] = chat_name

    try:
        response = requests.post(
            f"{API_BASE_URL}/chats",
            files=files,
            data=data,
            timeout=1800,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        return None

    return None


def send_message(
    chat_id: str,
    message: str,
    profile_label: str | None = None,
) -> str | None:
    payload: dict[str, Any] = {"message": message}
    if profile_label:
        payload["profile_id"] = profile_label

    try:
        response = requests.post(
            f"{API_BASE_URL}/chats/{chat_id}/messages",
            json=payload,
            timeout=1800,
        )
        if response.status_code == 200:
            return response.json().get("response", "")
    except Exception:
        return None

    return None
