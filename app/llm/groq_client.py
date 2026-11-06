"""Groq implementation of the provider-independent LLM interface."""

from typing import Any

from groq import Groq

from app.core.config import Settings


class GroqLLMClient:
    """Send chat-completion requests to Groq using the configured model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings.from_environment()
        if not self.settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Add your key to the project .env file."
            )

        self.client = Groq(api_key=self.settings.groq_api_key)

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        """Generate a response and translate the generic schema for Groq."""
        request: dict[str, Any] = {
            "model": self.settings.groq_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_completion_tokens": 800,
            "reasoning_effort": "low",
        }
        if response_schema is not None:
            request["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "smartdocs_answer",
                    "strict": True,
                    "schema": response_schema,
                },
            }
        completion = self.client.chat.completions.create(**request)

        answer = completion.choices[0].message.content
        if not answer:
            raise RuntimeError("Groq returned an empty answer")
        return answer.strip()
