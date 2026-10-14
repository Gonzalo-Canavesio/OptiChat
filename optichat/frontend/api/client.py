import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile

from optichat.frontend.state import ChatInfo

API_BASE_URL = "http://localhost:8000"


class BackendError(Exception):
    """The backend rejected or failed a request; the message is meant for the user."""


def _raise_for_error(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None
    raise BackendError(detail if isinstance(detail, str) else response.text[:300])


def _request(method: str, path: str, **kwargs) -> dict | list:
    try:
        response = requests.request(method, f"{API_BASE_URL}{path}", **kwargs)
    except requests.RequestException as e:
        raise BackendError(f"Could not reach the backend at {API_BASE_URL}: {e}") from e
    _raise_for_error(response)
    return response.json()


def fetch_chats() -> list[ChatInfo]:
    try:
        response = requests.get(f"{API_BASE_URL}/chats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


def fetch_chat(chat_id: str) -> ChatInfo | None:
    try:
        return _request("GET", f"/chats/{chat_id}", timeout=30)
    except BackendError:
        return None


def create_chat(
    model_files: list[UploadedFile],
    data_files: list[UploadedFile] | None,
    model_paper: UploadedFile | None,
    modeling_language: str,
    profile_label: str,
    solver: str,
    chat_name: str = "",
) -> ChatInfo:
    payload = {
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

    return _request(
        "POST", f"/chats/{chat_id}/messages", json=payload, timeout=1800
    )
