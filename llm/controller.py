import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import json
import time
from typing import List, Dict, Any, Optional
from openai import OpenAI, APIStatusError, APIConnectionError, APIResponseValidationError
from common.utils import extract_json_from_content
from common import config
import logging
logger = logging.getLogger(__name__)


def _format_exception_chain(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cur = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return " <- ".join(parts)


def _is_deepseek_v4_model(model: str) -> bool:
    return config.API_PROVIDER == "deepseek" and str(model).startswith("deepseek-v4")


def _add_deepseek_defaults(req: Dict[str, Any], model: str) -> None:
    if _is_deepseek_v4_model(model) and config.DEEPSEEK_THINKING_MODE == "disabled":
        req.setdefault("extra_body", {"thinking": {"type": "disabled"}})


class LLM:
    def __init__(self):
        self.client = OpenAI(api_key=config.API_KEY,
                             base_url=config.CHAT_BASE_URL,
                             timeout=config.LLM_TIMEOUT,
                             max_retries=config.LLM_CLIENT_MAX_RETRIES,
                             default_headers=config.OPENAI_COMPAT_DEFAULT_HEADERS)
        self.model = config.MODEL

    def chat_with_tool(
            self,
            *,
            messages: List[Dict[str, Any]],
            model: str = config.MODEL,
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[Any] = "auto",
            use_tool: bool = True,
            temperature: float = 0.0,
            top_p: float = 1.0,
            seed: Optional[int] = 66,
            max_retries: int = config.LLM_REQUEST_MAX_RETRIES,
            backoff: float = config.LLM_BACKOFF,
            **extra  # extra params, e.g. response_format
    ):
        """
        Robust wrapper around client.chat.completions.create:
        - catches 429/5xx/connection/validation errors with exponential backoff
        - supports passing tools / tool_choice
        - returns the OpenAI SDK ChatCompletion object
        """
        req = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        if config.API_PROVIDER not in {"deepseek", "ofox", "openrouter"} and seed is not None:
            req["seed"] = seed
        if use_tool:
            req["tools"] = tools
            req["tool_choice"] = tool_choice
        _add_deepseek_defaults(req, model)
        if extra:
            req.update(extra)

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp =  self.client.chat.completions.create(**req)
                return resp
            except APIStatusError as e:
                status = getattr(e, "status_code", None)
                text = getattr(getattr(e, "response", None), "text", "") or ""
                logger.warning(f"APIStatusError {status}: {text[:400]}")
                if status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                elif status == 400:
                    return "400"
                last_exc = e
                break


            except (APIConnectionError, APIResponseValidationError) as e:
                logger.warning(
                    "Connection/Validation error on attempt "
                    f"{attempt}/{max_retries}: {_format_exception_chain(e)}"
                )
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                last_exc = e
                break

            except Exception as e:
                logger.warning(f"Unexpected error: {repr(e)}", exc_info=True)
                if attempt < max_retries:
                    time.sleep(backoff ** attempt)
                    continue
                last_exc = e
                break

        if last_exc:
            raise last_exc

    def chat_text(
            self,
            *,
            messages: List[Dict[str, Any]],
            tools: Optional[List[Dict[str, Any]]] = None,
            tool_choice: Optional[Any] = "auto",
            model: str = config.MODEL,
            temperature: float = 0.0,
            **extra
    ) -> str:

        max_attempts = 3
        json_out = None

        for attempt in range(max_attempts):

            comp = self.chat_with_tool(
                messages=messages, model=model, tools=tools, tool_choice=tool_choice, use_tool=False,
                temperature=temperature, **extra
            )
            ch0 = comp.choices[0]
            msg = getattr(ch0, "message", None)

            if msg is not None:
                c = msg.content
                if isinstance(c, list):  # rich content structure
                    text = "".join(
                        getattr(p, "text", "") for p in c
                        if getattr(p, "type", "") == "text"
                    )
                else:
                    text = c or ""
            else:
                text = getattr(ch0, "text", "") or ""

            try:
                json_out = json.loads(text)
                break
            except json.JSONDecodeError:
                try:
                    json_out = extract_json_from_content(text)
                    break
                except (json.JSONDecodeError, ValueError) as e:
                    # log the error and keep retrying even if parsing fails
                    logger.warning(f"chat_text: failed to parse JSON on attempt {attempt}: {e}")
                    continue
        return json_out






