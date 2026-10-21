from pathlib import Path

import yaml

CREDENTIALS_FILE = Path(".llm_credentials.yaml")
PROFILES_FILE = Path(".llm_profiles.yaml")


def generate_valid_label(base_label: str, existing_labels: set[str]) -> str:
    final_label = base_label.strip()
    counter = 2

    while final_label in existing_labels:
        final_label = f"{base_label} ({counter})"
        counter += 1

    return final_label


def load_items(path: Path, key: str) -> list[dict]:
    if not path.is_file():
        return []

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        return []

    items = data.get(key, [])

    if not isinstance(items, list):
        return []

    return items


def save_items(path: Path, key: str, items: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump({key: items}, file)
