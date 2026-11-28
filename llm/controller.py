import json
import time

from openai import APIConnectionError, APIStatusError, OpenAI

from common import config
from common.utils import extract_json_from_content


class LLM:
    def __init__(self):
        self.client = OpenAI(
            api_key=config.API_KEY,
            base_url=config.CHAT_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,
        )

    def chat(self, messages, model=None, temperature=0.0):
        request = {
            "model": model or config.MODEL,
            "messages": messages,
            "temperature": temperature,
        }
        if config.API_PROVIDER == "deepseek" and config.DEEPSEEK_THINKING_MODE == "disabled":
            request["extra_body"] = {"thinking": {"type": "disabled"}}
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                return self.client.chat.completions.create(**request)
            except (APIConnectionError, APIStatusError):
                if attempt + 1 == config.LLM_MAX_RETRIES:
                    raise
                time.sleep(1.8 ** (attempt + 1))

    def chat_text(self, messages, model=None, temperature=0.0):
        response = self.chat(messages, model=model, temperature=temperature)
        content = response.choices[0].message.content or ""
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return extract_json_from_content(content)
