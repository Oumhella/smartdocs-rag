"""Groq implementation of the provider-independent LLM interface."""

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

    def generate(self, *, system_prompt: str, user_prompt: str) -> str:
        """Generate a response; keep provider-specific details in this class."""
        completion = self.client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=800,
            reasoning_effort="low",
        )

        answer = completion.choices[0].message.content
        if not answer:
            raise RuntimeError("Groq returned an empty answer")
        return answer.strip()
