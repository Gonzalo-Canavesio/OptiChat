from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx

VERIFICATION_TIMEOUT_SECONDS = 10


class ProviderVerificationError(Exception):
    """The provider's API key could not be verified; the message is user-facing."""


class APIKeyRejectedError(ProviderVerificationError):
    pass


class ProviderUnavailableError(ProviderVerificationError):
    pass


def _bearer_auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _openai_style_model_ids(payload: dict[str, Any]) -> list[str]:
    """Read the model ids of an OpenAI-style `{"data": [{"id": ...}]}` listing."""
    return [model["id"] for model in payload.get("data", [])]


def _is_key_rejected_by_status(response: httpx.Response) -> bool:
    """Whether the provider refused the key with 401 Unauthorized or 403 Forbidden."""
    return response.status_code in (401, 403)


@dataclass(frozen=True)
class ProviderSpec:
    """How to verify a provider's API key and list the models it can use."""

    name: str
    models_url: str
    auth_headers: Callable[[str], dict[str, str]]
    # Prefix LiteLLM needs to route the model; empty keeps the provider's bare id.
    litellm_prefix: str = ""
    parse_models: Callable[[dict[str, Any]], list[str]] = _openai_style_model_ids
    is_key_rejected: Callable[[httpx.Response], bool] = _is_key_rejected_by_status


# To support a new provider, add its spec here.
PROVIDER_SPECS = [
    ProviderSpec(
        name="OpenAI",
        models_url="https://api.openai.com/v1/models",
        auth_headers=_bearer_auth_headers,
    ),
    ProviderSpec(
        name="Anthropic",
        models_url="https://api.anthropic.com/v1/models?limit=1000",
        auth_headers=lambda api_key: {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        litellm_prefix="anthropic/",
    ),
    ProviderSpec(
        name="DeepSeek",
        models_url="https://api.deepseek.com/models",
        auth_headers=_bearer_auth_headers,
        litellm_prefix="deepseek/",
    ),
]


def get_available_provider_names() -> list[str]:
    """Providers the user can choose from, one per entry in PROVIDER_SPECS."""
    return [spec.name for spec in PROVIDER_SPECS]


def get_provider_spec(provider: str) -> ProviderSpec:
    """Return the spec registered under `provider`; raise ValueError if none."""
    for spec in PROVIDER_SPECS:
        if spec.name == provider:
            return spec
    raise ValueError(f"Unknown LLM provider '{provider}'.")


def fetch_provider_models(
    provider: str, api_key: str, client: httpx.Client | None = None
) -> list[str]:
    """Return the LiteLLM model ids `api_key` can use; raise if it can't be verified.

    Raises APIKeyRejectedError when the provider refuses the key, and
    ProviderUnavailableError when it can't tell (network error, rate limit,
    server error or an unexpected answer).
    """
    spec = get_provider_spec(provider)
    http = client or httpx.Client(timeout=VERIFICATION_TIMEOUT_SECONDS)
    try:
        response = http.get(spec.models_url, headers=spec.auth_headers(api_key))
    except httpx.HTTPError as e:
        raise ProviderUnavailableError(f"Could not reach {spec.name}.") from e
    finally:
        if client is None:
            http.close()

    if spec.is_key_rejected(response):
        raise APIKeyRejectedError(f"{spec.name} rejected the API key.")
    if not response.is_success:
        raise ProviderUnavailableError(
            f"{spec.name} answered with status {response.status_code}. "
            "Please try again later."
        )

    try:
        models = spec.parse_models(response.json())
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ProviderUnavailableError(
            f"{spec.name} returned an unexpected model list."
        ) from e
    return sorted(spec.litellm_prefix + model for model in models)
