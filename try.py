import os
import sys

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except ModuleNotFoundError as exc:
    missing = exc.name or "a required package"
    print(f"dependency failed: missing {missing}. Run `pip install -r requirements.txt` first.")
    sys.exit(2)

from common.openrouter import OPENROUTER_BASE_URL, get_openrouter_headers


load_dotenv()

CHAT_MODEL = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-large")
JUDGE_MODEL = os.getenv("OPENROUTER_JUDGE_MODEL", "openai/gpt-4o-mini")


def get_client() -> OpenAI:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key or api_key == "sk-or-v1-xxxxxxxx":
        raise RuntimeError("OPENROUTER_API_KEY is empty or still set to the placeholder in .env.")
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        default_headers=get_openrouter_headers(),
        timeout=60.0,
    )


def check_chat(client: OpenAI) -> None:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": "Reply with exactly: ok"}],
        temperature=0,
        max_tokens=8,
    )
    print(f"chat ok: {resp.choices[0].message.content!r}")


def check_embedding(client: OpenAI) -> None:
    resp = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=["connection test"],
    )
    print(f"embedding ok: dimension={len(resp.data[0].embedding)}")


def check_judge(client: OpenAI) -> None:
    resp = client.chat.completions.create(
        model=JUDGE_MODEL,
        messages=[{"role": "user", "content": "Return JSON only: {\"label\":\"CORRECT\"}"}],
        temperature=0,
        response_format={"type": "json_object"},
        max_tokens=32,
    )
    print(f"judge ok: {resp.choices[0].message.content!r}")


def main() -> int:
    try:
        client = get_client()
    except Exception as exc:
        print(f"config failed: {exc}")
        return 2

    checks = [
        ("chat", check_chat),
        ("embedding", check_embedding),
        ("judge", check_judge),
    ]
    failed = False
    for name, check in checks:
        try:
            check(client)
        except Exception as exc:
            failed = True
            print(f"{name} failed: {type(exc).__name__}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
