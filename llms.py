from pathlib import Path

import yaml
from pydantic import BaseModel, Field

LLM_PROVIDERS = [
    {"type": "OpenAI", "env_api_key_litellm": "OPENAI_API_KEY"},
    {"type": "Anthropic", "env_api_key_litellm": "ANTHROPIC_API_KEY"},
    {"type": "Google Gemini", "env_api_key_litellm": "GOOGLE_GEMINI_API_KEY"},
    {"type": "DeepSeek", "env_api_key_litellm": "DEEPSEEK_API_KEY"},
]


class LLMProviderConfig(BaseModel):
    label: str | None = Field(
        None, description="A user-friendly label for the LLM provider."
    )
    type: str = Field(
        ..., description="The type of the LLM provider (e.g., OpenAI, Anthropic)."
    )
    api_key: str = Field(..., description="The API key for the LLM provider.")


def get_available_types_llm_providers() -> list[str]:
    return [provider["type"] for provider in LLM_PROVIDERS]


def _get_configured_llm_providers_from_file() -> list[LLMProviderConfig]:
    config_path = Path("llm_providers.yaml")

    if not config_path.is_file():
        return []

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

        if not isinstance(data, dict):
            return []

        providers = data.get("providers", [])

        if not isinstance(providers, list):
            return []

        return [LLMProviderConfig(**provider) for provider in providers]


def get_configured_llm_providers() -> list[LLMProviderConfig]:
    return _get_configured_llm_providers_from_file()


def _generate_unique_label(base_label: str, existing_labels: set[str]) -> str:
    final_label = base_label
    counter = 2

    while final_label in existing_labels:
        final_label = f"{base_label} ({counter})"
        counter += 1

    return final_label


def add_llm_provider(provider: LLMProviderConfig):
    providers = _get_configured_llm_providers_from_file()
    existing_labels = {
        p.label for p in providers if p.type == provider.type and p.label is not None
    }
    base_label = provider.label if provider.label else provider.type
    provider.label = _generate_unique_label(base_label, existing_labels)
    providers.append(provider)

    config_path = Path("llm_providers.yaml")
    with config_path.open("w", encoding="utf-8") as file:
        serializable_providers = [p.model_dump() for p in providers]
        yaml.safe_dump({"providers": serializable_providers}, file)
