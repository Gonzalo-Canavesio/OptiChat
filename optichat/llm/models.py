from pydantic import BaseModel, Field


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
