import os
from typing import Dict, Any, Optional

class LLMProvider:
    """
    LLM Interface supporting Google Gemini API, OpenAI API, or deterministic fallback mode.
    Ensures zero reliance on mandatory API keys while enabling LLM enrichment when available.
    """

    def __init__(self, provider: str = "auto"):
        self.gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self.openai_key = os.environ.get("OPENAI_API_KEY")
        self.provider = provider

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> Optional[str]:
        # Try Gemini if key available
        if (self.provider in ["auto", "gemini"]) and self.gemini_key:
            try:
                from google import genai
                client = genai.Client(api_key=self.gemini_key)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config={"system_instruction": system_instruction} if system_instruction else None
                )
                return response.text
            except Exception as e:
                print(f"Gemini API invocation failed: {e}")

        # Try OpenAI if key available
        if (self.provider in ["auto", "openai"]) and self.openai_key:
            try:
                import openai
                client = openai.OpenAI(api_key=self.openai_key)
                messages = []
                if system_instruction:
                    messages.append({"role": "system", "content": system_instruction})
                messages.append({"role": "user", "content": prompt})
                res = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
                return res.choices[0].message.content
            except Exception as e:
                print(f"OpenAI API invocation failed: {e}")

        # Return None to signal deterministic engine synthesis
        return None
