import os
from pathlib import Path
from unittest.mock import patch

import yaml
from litellm import litellm
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


class AgentConfig(BaseModel):
    provider_label: str = Field(
        ..., description="The label of the LLM provider to use for the agent."
    )
    llm_model: str = Field(..., description="The LLM model associated with the agent.")
    temperature: float | None = Field(
        default=None,
        description="Optional temperature setting for the agent's LLM model.",
        ge=0.0,
        le=1.0,
    )
    max_tokens: int | None = Field(
        default=None,
        description="Optional maximum token limit for the agent's LLM model.",
        ge=1,
    )


class ProfileConfig(BaseModel):
    label: str = Field(..., description="A user-friendly label for the profile.")
    root_agent: AgentConfig = Field(
        ..., description="Configuration for the Root Agent."
    )
    expert_agent: AgentConfig = Field(
        ..., description="Configuration for the Expert Agent."
    )
    illustrator_agent: AgentConfig = Field(
        ..., description="Configuration for the Illustrator Agent."
    )
    generator_agent: AgentConfig = Field(
        ..., description="Configuration for the Generator Agent."
    )


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


def _get_env_var_name_for_provider(provider_name: str | None) -> str | None:
    for provider in LLM_PROVIDERS:
        if provider["type"] == provider_name:
            return provider["env_api_key_litellm"]
    return None


def get_available_llm_models(provider: LLMProviderConfig) -> list[str]:
    temp_env_var = {_get_env_var_name_for_provider(provider.type): provider.api_key}
    with patch.dict(os.environ, temp_env_var):
        return litellm.get_valid_models()


def _get_llm_profiles_from_file() -> list[ProfileConfig]:
    config_path = Path("llm_profiles.yaml")

    if not config_path.is_file():
        return []

    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

        if not isinstance(data, dict):
            return []

        profiles = data.get("profiles", [])

        if not isinstance(profiles, list):
            return []

        return [ProfileConfig(**profile) for profile in profiles]


def get_llm_profiles():
    return _get_llm_profiles_from_file()


def add_llm_profile(profile: ProfileConfig):
    profiles = _get_llm_profiles_from_file()
    existing_labels = {p.label for p in profiles if p.label is not None}
    profile.label = _generate_unique_label(profile.label, existing_labels)
    profiles.append(profile)

    config_path = Path("llm_profiles.yaml")
    with config_path.open("w", encoding="utf-8") as file:
        serializable_profiles = [p.model_dump() for p in profiles]
        yaml.safe_dump({"profiles": serializable_profiles}, file)


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
