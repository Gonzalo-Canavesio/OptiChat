import json

import httpx

from optichat.llm.providers.anthropic import Anthropic
from optichat.llm.providers.base import (
    APIKeyRejectedError,
    Provider,
    ProviderModel,
    ProviderUnavailableError,
)
from optichat.llm.providers.deepseek import DeepSeek
from optichat.llm.providers.ollama import Ollama
from optichat.llm.providers.openai import OpenAI

VERIFICATION_TIMEOUT_SECONDS = 10

# To support a new provider, add a module with its Provider subclass
# and list an instance here.
PROVIDERS: list[Provider] = [OpenAI(), Anthropic(), DeepSeek(), Ollama()]


def get_available_provider_names() -> list[str]:
    """Providers the user can choose from, one per entry in PROVIDERS."""
    return [provider.name for provider in PROVIDERS]


def get_provider(name: str) -> Provider:
    """Return the provider registered under `name`; raise ValueError if none."""
    for provider in PROVIDERS:
        if provider.name == name:
            return provider
    raise ValueError(f"Unknown LLM provider '{name}'.")


def fetch_provider_models(
    provider: str, api_key: str | None, transport: httpx.BaseTransport | None = None
) -> list[ProviderModel]:
    """Return the provider's models sorted by id; raise if access can't be verified.

    Raises APIKeyRejectedError when the provider refuses the key, and
    ProviderUnavailableError when it can't tell (network error, rate limit,
    server error or an unexpected answer). A 401 or 403 only means the key was
    rejected when there was a key to send.
    """
    llm_provider = get_provider(provider)
    name = llm_provider.name
    with httpx.Client(
        timeout=VERIFICATION_TIMEOUT_SECONDS,
        transport=transport,
        event_hooks={"response": [_raise_for_status]},
    ) as http:
        try:
            models = llm_provider.list_models(http, api_key)
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            if api_key and status_code in (401, 403):
                raise APIKeyRejectedError(f"{name} rejected the API key.") from e
            raise ProviderUnavailableError(
                f"{name} answered with status {status_code}. Please try again later."
            ) from e
        except httpx.HTTPError as e:
            raise ProviderUnavailableError(f"Could not reach {name}.") from e
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as e:
            raise ProviderUnavailableError(
                f"{name} returned an unexpected answer."
            ) from e
    return sorted(models, key=lambda model: model.id)


def _raise_for_status(response: httpx.Response) -> None:
    # So every 4xx/5xx raises even when a provider doesn't check it.
    response.raise_for_status()
