import pydantic

"""Text processing utilities."""


class PromptConfig(pydantic.BaseModel):
    """Prompt config."""

    model_config = pydantic.ConfigDict(extra="forbid")

    system_prompt: str = pydantic.Field(default="", description="System prompt")
    user_prompt: str = pydantic.Field(default="", description="User prompt")
