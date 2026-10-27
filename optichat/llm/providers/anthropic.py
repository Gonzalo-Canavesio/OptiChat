import httpx

from optichat.llm.providers.base import Provider, ProviderModel
from optichat.llm.providers.litellm_map import litellm_accepts_pdf
from optichat.llm.providers.openai_style import list_openai_style_models


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}


class Anthropic(Provider):
    name = "Anthropic"
    api_key_required = True

    def list_models(
        self, http: httpx.Client, api_key: str | None
    ) -> list[ProviderModel]:
        return list_openai_style_models(
            http,
            self.name,
            "https://api.anthropic.com/v1/models?limit=1000",
            api_key,
            _auth_headers,
            litellm_prefix="anthropic/",
        )

    def accepts_pdf(self, model: str) -> bool:
        return litellm_accepts_pdf(model)
