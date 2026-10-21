LLM_PROVIDERS = [
    {"name": "OpenAI", "env_api_key_litellm": "OPENAI_API_KEY"},
    {"name": "Anthropic", "env_api_key_litellm": "ANTHROPIC_API_KEY"},
    {"name": "DeepSeek", "env_api_key_litellm": "DEEPSEEK_API_KEY"},
]


def get_available_provider_names() -> list[str]:
    """Providers the user can choose from, one per entry in LLM_PROVIDERS."""
    return [provider["name"] for provider in LLM_PROVIDERS]


def get_env_var_name_for_provider(provider_name: str | None) -> str | None:
    for provider in LLM_PROVIDERS:
        if provider["name"] == provider_name:
            return provider["env_api_key_litellm"]
    return None
