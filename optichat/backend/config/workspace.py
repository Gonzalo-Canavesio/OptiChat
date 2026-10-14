import os
from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_workspace: ContextVar[Path | None] = ContextVar("optichat_workspace", default=None)


@contextmanager
def use_workspace(path: str | Path) -> Generator[Path]:
    root = Path(path).resolve()
    token = _workspace.set(root)
    try:
        yield root
    finally:
        _workspace.reset(token)


def workspace_root() -> Path:
    return _workspace.get() or Path(os.getcwd()) / "tmp"


def model_objects_dir() -> Path:
    return workspace_root() / "model_objects"


def model_data_dir() -> Path:
    return model_objects_dir() / "data"


def metadata_path() -> Path:
    return model_objects_dir() / "metadata.json"


def generated_papers_dir() -> Path:
    return model_objects_dir() / "generated_papers"


def robust_dir() -> Path:
    return workspace_root() / "robust"


def inf_detail_dir() -> Path:
    return workspace_root() / "inf_detail"


def iis_dir(version: str) -> Path:
    return workspace_root() / "iis" / version


def timing_dir() -> Path:
    return workspace_root() / "timing"
