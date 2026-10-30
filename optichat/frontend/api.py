from typing import Any

import httpx
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.frontend.state import ChatInfo

API_BASE_URL = "http://localhost:8000"

_client = httpx.Client(base_url=API_BASE_URL)


class BackendError(Exception):
    """The backend rejected or failed a request; the message is meant for the user."""


def _raise_for_error(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    raise BackendError(detail if isinstance(detail, str) else response.text[:300])


def _request(method: str, path: str, **kwargs) -> Any:
    try:
        response = _client.request(method, path, **kwargs)
    except httpx.HTTPError as e:
        raise BackendError(f"Could not reach the backend at {API_BASE_URL}: {e}") from e
    _raise_for_error(response)
    try:
        return response.json()
    except ValueError as e:
        raise BackendError("The backend sent a response that is not valid JSON.") from e


def fetch_chats() -> list[ChatInfo]:
    return _request("GET", "/chats", timeout=10)


def fetch_chat(chat_id: str) -> ChatInfo:
    return _request("GET", f"/chats/{chat_id}", timeout=30)


def create_chat(
    model_file: UploadedFile,
    data_files: list[UploadedFile] | None,
    model_paper: UploadedFile | None,
    profile_label: str,
    solver: str,
    chat_name: str = "",
) -> ChatInfo:
    payload = {
        "profile_label": profile_label,
        "solver": solver,
        "chat_name": chat_name,
    }

    files = [("model_file", (model_file.name, model_file.getvalue(), "text/x-python"))]

    for data_file in data_files or []:
        content_type = data_file.type or "application/octet-stream"
        files.append(
            ("data_files", (data_file.name, data_file.getvalue(), content_type))
        )

    if model_paper:
        content_type = model_paper.type or "application/octet-stream"
        files.append(
            ("model_paper", (model_paper.name, model_paper.getvalue(), content_type))
        )

    return _request("POST", "/chats", data=payload, files=files, timeout=900)


def send_message(
    chat_id: str, message: str, profile_label: str, solver: str
) -> ChatInfo:
    payload = {
        "message": message,
        "profile_label": profile_label,
        "solver": solver,
    }

    return _request("POST", f"/chats/{chat_id}/messages", json=payload, timeout=1800)
