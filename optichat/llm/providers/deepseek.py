import httpx

from optichat.llm.providers.base import Provider, ProviderModel
from optichat.llm.providers.litellm_map import litellm_accepts_pdf
from optichat.llm.providers.openai_style import (
    bearer_auth_headers,
    list_openai_style_models,
)


class DeepSeek(Provider):
    name = "DeepSeek"
    api_key_required = True

    def list_models(
        self, http: httpx.Client, api_key: str | None
    ) -> list[ProviderModel]:
        return list_openai_style_models(
            http,
            self.name,
            "https://api.deepseek.com/models",
            api_key,
            bearer_auth_headers,
            litellm_prefix="deepseek/",
        )

    def accepts_pdf(self, model: str) -> bool:
        return litellm_accepts_pdf(model)
