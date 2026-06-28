import os
import sys
import argparse

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(f"dependency failed: missing {missing}. Run `pip install -r requirements.txt` first.")
    sys.exit(2)

from common.openrouter import OPENROUTER_BASE_URL, get_openrouter_headers


load_dotenv()

OPENROUTER_CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")
OPENROUTER_EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
OPENROUTER_JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_CHAT_MODEL = os.getenv("DEEPSEEK_CHAT_MODEL", "deepseek-chat")
DEEPSEEK_JUDGE_MODEL = os.getenv("DEEPSEEK_JUDGE_MODEL", "deepseek-v4-flash")
DEEPSEEK_THINKING_MODE = os.getenv("DEEPSEEK_THINKING_MODE", "disabled").lower()


def describe_exception(exc: Exception) -> str:
    parts = [f"{type(exc).__name__}: {exc}"]
    seen = {id(exc)}
    cur = getattr(exc, "__cause__", None) or getattr(exc, "__context__", None)
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    return " <- ".join(parts)


def get_openrouter_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "sk-or-v1-xxxxxxxx":
        raise RuntimeError("OPENROUTER_API_KEY is empty or still set to the placeholder in .env.")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=get_openrouter_headers(),
        timeout=60.0,
    )


def get_deepseek_client() -> OpenAI:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or api_key == "sk-xxxxxxxx":
        raise RuntimeError("DEEPSEEK_API_KEY is empty or still set to the placeholder in .env.")
    return OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
        timeout=float(os.getenv("DEEPSEEK_SMOKE_TIMEOUT", "120")),
        max_retries=int(os.getenv("DEEPSEEK_SMOKE_RETRIES", "3")),
    )


def maybe_deepseek_extra_body(model: str) -> dict:
    if model.startswith("deepseek-v4") and DEEPSEEK_THINKING_MODE == "disabled":
        return {"extra_body": {"thinking": {"type": "disabled"}}}
    return {}


def check_chat(client: OpenAI, model: str, provider: str) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=8,
        **(maybe_deepseek_extra_body(model) if provider == "deepseek" else {}),
    )
    print(f"chat ok: {resp.choices[0].message.content!r}")


def check_embedding(client: OpenAI, model: str) -> None:
    resp = client.embeddings.create(
        model=model,
        input=["connection test"],
    )
    print(f"embedding ok: dimension={len(resp.data[0].embedding)}")


def check_judge(client: OpenAI, model: str, provider: str) -> None:
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Return JSON only: {\"label\":\"CORRECT\"}"}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=32,
        **(maybe_deepseek_extra_body(model) if provider == "deepseek" else {}),
    )
    print(f"judge ok: {resp.choices[0].message.content!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Provider connectivity smoke test.")
    parser.add_argument("--provider", choices=["openrouter", "deepseek"], default="openrouter")
    parser.add_argument("--chat-model", default=None)
    parser.add_argument("--judge-model", default=None)
    args = parser.parse_args()

    try:
        if args.provider == "deepseek":
            client = get_deepseek_client()
            checks = [
                ("chat", lambda c: check_chat(c, args.chat_model or DEEPSEEK_CHAT_MODEL, "deepseek")),
                ("judge", lambda c: check_judge(c, args.judge_model or DEEPSEEK_JUDGE_MODEL, "deepseek")),
            ]
        else:
            client = get_openrouter_client()
            checks = [
                ("chat", lambda c: check_chat(c, args.chat_model or OPENROUTER_CHAT_MODEL, "openrouter")),
                ("embedding", lambda c: check_embedding(c, OPENROUTER_EMBEDDING_MODEL)),
                ("judge", lambda c: check_judge(c, args.judge_model or OPENROUTER_JUDGE_MODEL, "openrouter")),
            ]
    except Exception as exc:
        print(f"config failed: {describe_exception(exc)}")
        return 2

    failed = False
    for name, check in checks:
        try:
            check(client)
        except Exception as exc:
            failed = True
            print(f"{name} failed: {describe_exception(exc)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
