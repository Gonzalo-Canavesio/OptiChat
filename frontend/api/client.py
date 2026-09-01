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
    model_files: list[UploadedFile],
    data_files: list[UploadedFile] | None,
    modeling_language: str,
    profile_label: str,
    solver: str,
    chat_name: str = "",
) -> ChatInfo | None:
    data = {
        "profile_label": profile_label,
        "modeling_language": modeling_language,
        "solver": "gurobi" if modeling_language == "gurobipy" else solver,
        "chat_name": chat_name,
    }

    files = []
    for model_file in model_files:
        files.append(
            ("model_files", (model_file.name, model_file.getvalue(), "text/x-python"))
        )

    for data_file in data_files or []:
        content_type = data_file.type or "application/octet-stream"
        files.append(
            ("data_files", (data_file.name, data_file.getvalue(), content_type))
        )

    try:
        response = requests.post(
            f"{API_BASE_URL}/chat/create",
            data=data,
            files=files,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
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
