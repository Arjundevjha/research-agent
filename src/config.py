"""
Central LLM configuration.

Supports two free providers:
  1. Google AI Studio (recommended) — 1,500 free requests/day
  2. OpenRouter free models — 50 requests/day (or 1,000 with $10 credit)

Set your provider via the LLM_PROVIDER env var ("google" or "openrouter").
"""

import os
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()


def get_llm() -> LLM:
    """Return the shared LLM instance based on the configured provider."""
    provider = os.getenv("LLM_PROVIDER", "google").lower()

    if provider == "google":
        return _google_llm()
    elif provider == "openrouter":
        return _openrouter_llm()
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{provider}'. Use 'google' or 'openrouter'."
        )


def _google_llm() -> LLM:
    """Google AI Studio — free tier, 1,500 RPD, large context."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is missing. Get a free key at:\n"
            "  https://aistudio.google.com/apikey\n"
            "Then add it to your .env file."
        )

    model = os.getenv("GOOGLE_MODEL", "gemini/gemini-2.5-flash")

    return LLM(
        model=model,
        api_key=api_key,
        temperature=0.3,
        max_retries=5,
    )


def _openrouter_llm() -> LLM:
    """OpenRouter free models — 50 RPD free, 1,000 with $10 credit."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file."
        )

    # Free models ordered by current availability
    free_models = [
        "openrouter/google/gemma-4-31b-it:free",
        "openrouter/openai/gpt-oss-120b:free",
        "openrouter/nvidia/nemotron-3-super-120b-a12b:free",
        "openrouter/deepseek/deepseek-v4-flash:free",
        "openrouter/minimax/minimax-m2.5:free",
    ]

    model = os.getenv("FREE_MODEL", free_models[0])

    return LLM(
        model=model,
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        temperature=0.3,
        max_retries=5,
    )
