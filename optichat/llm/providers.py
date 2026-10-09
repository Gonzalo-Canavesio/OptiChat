from optichat.llm.models import LLMProviderConfig
from optichat.llm.profiles import (
    remove_provider_from_profiles,
    update_provider_label_in_profiles,
)
from optichat.llm.store import (
    PROVIDERS_FILE,
    generate_valid_label,
    load_items,
    save_items,
)

LLM_PROVIDERS = [
    {"type": "OpenAI", "env_api_key_litellm": "OPENAI_API_KEY"},
    {"type": "Anthropic", "env_api_key_litellm": "ANTHROPIC_API_KEY"},
    {"type": "Google Gemini", "env_api_key_litellm": "GOOGLE_GEMINI_API_KEY"},
    {"type": "DeepSeek", "env_api_key_litellm": "DEEPSEEK_API_KEY"},
]


def _get_configured_llm_providers_from_file() -> list[LLMProviderConfig]:
    return [LLMProviderConfig(**p) for p in load_items(PROVIDERS_FILE, "providers")]


def _save_configured_llm_providers_to_file(providers: list[LLMProviderConfig]):
    save_items(PROVIDERS_FILE, "providers", [p.model_dump() for p in providers])


def _get_valid_label_for_provider(provider: LLMProviderConfig) -> str:
    providers = _get_configured_llm_providers_from_file()
    existing_labels = {p.label for p in providers if p.label is not None}
    base_label = provider.label if provider.label else provider.type
    return generate_valid_label(base_label, existing_labels)


def get_available_types_llm_providers() -> list[str]:
    return [provider["type"] for provider in LLM_PROVIDERS]


def get_env_var_name_for_provider(provider_name: str | None) -> str | None:
    for provider in LLM_PROVIDERS:
        if provider["type"] == provider_name:
            return provider["env_api_key_litellm"]
    return None


def get_configured_llm_providers() -> list[LLMProviderConfig]:
    return _get_configured_llm_providers_from_file()


def get_llm_provider_by_label(label: str | None) -> LLMProviderConfig | None:
    return next(
        (p for p in _get_configured_llm_providers_from_file() if p.label == label),
        None,
    )


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

    update_provider_label_in_profiles(existing_provider.label, updated_provider.label)


def remove_llm_provider(provider: LLMProviderConfig):
    if not provider.label:
        raise ValueError("Existing provider must have a label.")

    providers = _get_configured_llm_providers_from_file()
    providers = [p for p in providers if p != provider]

    _save_configured_llm_providers_to_file(providers)

    remove_provider_from_profiles(provider.label)
