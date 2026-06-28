import os
from typing import Dict, Optional


OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")


def get_openrouter_headers() -> Optional[Dict[str, str]]:
    """Optional headers recommended by OpenRouter for app attribution."""
    headers = {}
    referer = os.getenv("OPENROUTER_HTTP_REFERER") or os.getenv("OPENROUTER_SITE_URL")
    title = os.getenv("OPENROUTER_APP_NAME", "MRAgent")
    if referer:
        headers["HTTP-Referer"] = referer
    if title:
        headers["X-Title"] = title
    return headers or None
