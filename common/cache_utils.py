import json
from pathlib import Path


def load_jsonl_records(path):
    """Load non-empty JSONL records and identify a truncated cache clearly."""
    cache_path = Path(path)
    if not cache_path.exists():
        return []

    records = []
    with cache_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            payload = line.strip()
            if not payload:
                continue
            try:
                records.append(json.loads(payload))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in cache {cache_path} at line "
                    f"{line_number}: {exc.msg}"
                ) from exc
    return records


def validate_rewrite_cache_prefix(records, expected_session_ids, path):
    """Require a rewrite cache to be an ordered prefix of sample sessions."""
    expected = [str(session_id) for session_id in expected_session_ids]
    if len(records) > len(expected):
        raise ValueError(
            f"rewrite cache {path} has {len(records)} records but the sample "
            f"has only {len(expected)} sessions"
        )

    for index, record in enumerate(records):
        if not isinstance(record, dict) or len(record) != 1:
            raise ValueError(
                f"rewrite cache {path} record {index + 1} must contain "
                "exactly one session"
            )
        actual_session_id = str(next(iter(record)))
        expected_session_id = expected[index]
        if actual_session_id != expected_session_id:
            raise ValueError(
                f"rewrite cache {path} record {index + 1} contains "
                f"{actual_session_id!r}; expected {expected_session_id!r}. "
                "The cache is not a resumable prefix of this sample."
            )


def index_keyword_sentences(records):
    """Index keyword rows by rewrite sentence ID instead of JSONL position."""
    indexed = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        for sentence in record.get("sentence") or []:
            if not isinstance(sentence, dict):
                continue
            sentence_id = sentence.get("sentence_id")
            if isinstance(sentence_id, str) and sentence_id:
                indexed[sentence_id] = sentence
    return indexed
