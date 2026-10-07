import os
from unittest.mock import patch

from litellm import litellm

from optichat.llm.models import LLMProviderConfig
from optichat.llm.providers import get_env_var_name_for_provider


def get_available_llm_models(provider: LLMProviderConfig) -> list[str]:
    temp_env_var = {get_env_var_name_for_provider(provider.type): provider.api_key}
    with patch.dict(os.environ, temp_env_var):
        return litellm.get_valid_models(check_provider_endpoint=True)


def valid_key_for_model(provider: LLMProviderConfig, model: str) -> bool:
    return litellm.check_valid_key(model=model, api_key=provider.api_key)
