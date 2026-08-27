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
    provider_label: str | None = Field(
        ..., description="The label of the LLM provider to use for the agent."
    )
    llm_model: str | None = Field(
        ..., description="The LLM model associated with the agent."
    )
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

    @property
    def enabled(self) -> bool:
        return bool(self.provider_label and self.llm_model)


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

    @property
    def enabled(self) -> bool:
        return (
            self.root_agent.enabled
            and self.expert_agent.enabled
            and self.illustrator_agent.enabled
            and self.generator_agent.enabled
        )


def _generate_valid_label(base_label: str, existing_labels: set[str]) -> str:
    final_label = base_label.strip()
    counter = 2

    while final_label in existing_labels:
        final_label = f"{base_label} ({counter})"
        counter += 1

    return final_label


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


def _save_configured_llm_providers_to_file(providers: list[LLMProviderConfig]):
    config_path = Path("llm_providers.yaml")
    with config_path.open("w", encoding="utf-8") as file:
        serializable_providers = [p.model_dump() for p in providers]
        yaml.safe_dump({"providers": serializable_providers}, file)


def _get_valid_label_for_provider(provider: LLMProviderConfig) -> str:
    providers = _get_configured_llm_providers_from_file()
    existing_labels = {p.label for p in providers if p.label is not None}
    base_label = provider.label if provider.label else provider.type
    return _generate_valid_label(base_label, existing_labels)


def get_available_types_llm_providers() -> list[str]:
    return [provider["type"] for provider in LLM_PROVIDERS]


def get_configured_llm_providers() -> list[LLMProviderConfig]:
    return _get_configured_llm_providers_from_file()


def add_llm_provider(provider: LLMProviderConfig):
    provider.label = _get_valid_label_for_provider(provider)

    providers = _get_configured_llm_providers_from_file()
    providers.append(provider)

    _save_configured_llm_providers_to_file(providers)


def update_llm_provider(
    existing_provider: LLMProviderConfig, updated_provider: LLMProviderConfig
):
    if not existing_provider.label:
        raise ValueError("Existing provider must have a label.")

    if not updated_provider.label or existing_provider.label != updated_provider.label:
        updated_provider.label = _get_valid_label_for_provider(updated_provider)

    providers = _get_configured_llm_providers_from_file()
    for i, provider in enumerate(providers):
        if provider.label == existing_provider.label:
            providers[i] = updated_provider
            break

    _save_configured_llm_providers_to_file(providers)

    _update_provider_label_in_profiles(existing_provider.label, updated_provider.label)


def remove_llm_provider(provider: LLMProviderConfig):
    if not provider.label:
        raise ValueError("Existing provider must have a label.")

    providers = _get_configured_llm_providers_from_file()
    providers = [p for p in providers if p != provider]

    _save_configured_llm_providers_to_file(providers)

    _remove_provider_from_profiles(provider.label)


def _get_env_var_name_for_provider(provider_name: str | None) -> str | None:
    for provider in LLM_PROVIDERS:
        if provider["type"] == provider_name:
            return provider["env_api_key_litellm"]
    return None


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


def _save_configured_llm_profiles_to_file(profiles: list[ProfileConfig]):
    config_path = Path("llm_profiles.yaml")
    with config_path.open("w", encoding="utf-8") as file:
        serializable_profiles = [p.model_dump() for p in profiles]
        yaml.safe_dump({"profiles": serializable_profiles}, file)


def _update_provider_label_in_profiles(old_label: str, new_label: str):
    profiles = _get_llm_profiles_from_file()
    for profile in profiles:
        if profile.root_agent.provider_label == old_label:
            profile.root_agent.provider_label = new_label
        if profile.expert_agent.provider_label == old_label:
            profile.expert_agent.provider_label = new_label
        if profile.illustrator_agent.provider_label == old_label:
            profile.illustrator_agent.provider_label = new_label
        if profile.generator_agent.provider_label == old_label:
            profile.generator_agent.provider_label = new_label

    _save_configured_llm_profiles_to_file(profiles)


def _remove_provider_from_profiles(provider_label: str):
    profiles = _get_llm_profiles_from_file()
    for profile in profiles:
        if profile.root_agent.provider_label == provider_label:
            profile.root_agent.provider_label = ""
            profile.root_agent.llm_model = ""
        if profile.expert_agent.provider_label == provider_label:
            profile.expert_agent.provider_label = ""
            profile.expert_agent.llm_model = ""
        if profile.illustrator_agent.provider_label == provider_label:
            profile.illustrator_agent.provider_label = ""
            profile.illustrator_agent.llm_model = ""
        if profile.generator_agent.provider_label == provider_label:
            profile.generator_agent.provider_label = ""
            profile.generator_agent.llm_model = ""

    _save_configured_llm_profiles_to_file(profiles)


def get_available_llm_models(provider: LLMProviderConfig) -> list[str]:
    temp_env_var = {_get_env_var_name_for_provider(provider.type): provider.api_key}
    with patch.dict(os.environ, temp_env_var):
        return litellm.get_valid_models(check_provider_endpoint=True)


def valid_key_for_model(provider: LLMProviderConfig, model: str) -> bool:
    return litellm.check_valid_key(model=model, api_key=provider.api_key)


def get_llm_profiles() -> list[ProfileConfig]:
    return _get_llm_profiles_from_file()


def _get_valid_label_for_profile(profile: ProfileConfig) -> str:
    profiles = _get_llm_profiles_from_file()
    existing_labels = {p.label for p in profiles if p.label is not None}
    return _generate_valid_label(profile.label, existing_labels)


def add_llm_profile(profile: ProfileConfig):
    profile.label = _get_valid_label_for_profile(profile)

    profiles = _get_llm_profiles_from_file()
    profiles.append(profile)

    _save_configured_llm_profiles_to_file(profiles)


def update_llm_profile(existing_profile: ProfileConfig, updated_profile: ProfileConfig):
    if existing_profile.label != updated_profile.label:
        updated_profile.label = _get_valid_label_for_profile(updated_profile)

    profiles = _get_llm_profiles_from_file()
    for i, profile in enumerate(profiles):
        if profile.label == existing_profile.label:
            profiles[i] = updated_profile
            break

    _save_configured_llm_profiles_to_file(profiles)


def remove_llm_profile(profile: ProfileConfig):
    profiles = _get_llm_profiles_from_file()
    profiles = [p for p in profiles if p != profile]

    _save_configured_llm_profiles_to_file(profiles)


def get_llm_profiles_using_provider(provider: LLMProviderConfig) -> list[ProfileConfig]:
    profiles = _get_llm_profiles_from_file()
    return [
        profile
        for profile in profiles
        if profile.root_agent.provider_label == provider.label
        or profile.expert_agent.provider_label == provider.label
        or profile.illustrator_agent.provider_label == provider.label
        or profile.generator_agent.provider_label == provider.label
    ]
