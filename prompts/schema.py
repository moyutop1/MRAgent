
SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Conversation Schema",
  "type": "object",
  "required": ["conversation_time", "sentence", "topics", "personal_sentences"],
  "properties": {
    "conversation_time": {
      "type": "string",
      "format": "date",
      "description": "YYYY-MM-DD"
    },
    "sentence": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "id", "text", "tag", "origin", "topic", "time",
          "memory_types", "persistence"
        ],
        "properties": {
          "id": {
            "type": "string",
            "pattern": "^D\\d+:\\d+-\\d+$",
          },
          "text": {
            "type": "string",
            "minLength": 1
          },
          "tag": {
            "type": "string",
          },
          "origin": {
            "type": "string",
            "pattern": "^D\\d+:\\d+(,\\s*D\\d+:\\d+)*$",
          },
          "topic": {
            "type": "array",
          },
          "time": {
            "type": "string",
            "format": "date",
            "description": "YYYY-MM-DD"
          },
          "memory_types": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "uniqueItems": True,
            "items": {
              "type": "string",
              "enum": [
                "event_action",
                "state_opinion",
                "profile_preference",
                "relation_social",
                "fact_background"
              ]
            }
          },
          "persistence": {
            "type": "string",
            "enum": ["transient", "episodic", "durable", "unknown"]
          }
        }
      }
    },
    "topics": {
      "type": "object",
    },
    "personal_sentences": {
      "type": "array",

    }
  }
}


KEY_SCHEMA = {
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "Mini Conversation Schema",
  "type": "object",
  "required": ["sentence"],
  "properties": {
    "sentence": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["sentence_id", "keyword"],
        "properties": {
          "sentence_id": {
            "type": "string",
            "pattern": "^D\\d+:\\d+-\\d+$",
          },
          "keyword": {
            "type": "array",
            #"minItems": 1,
            "items": { "type": "string", "minLength": 1 },
          }
        }
      }
    }
  }
}

import re, json, csv, os
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set

ID_RE = re.compile(r'^D\d+:\d+-\d+$')
ORIGIN_RE = re.compile(r'^D\d+:\d+(,\s*D\d+:\d+)*$')
DIA_EXTRACT_RE = re.compile(r'dia_id\s*:\s*(D\d+:\d+)', re.IGNORECASE)


def check_rewrite_json(text, dialogue_text):
  from jsonschema import Draft202012Validator, ValidationError
  import re
  schema = SCHEMA

  validator = Draft202012Validator(schema)
  ID_RE = re.compile(r'^D\d+:\d+-\d+$')
  ORIGIN_RE = re.compile(r'^D\d+:\d+(,\s*D\d+:\d+)*$')
  DIA_EXTRACT_RE = re.compile(r'dia_id\s*:\s*(D\d+:\d+)', re.IGNORECASE)

  # Step 1: Schema validation
  try:
    validator.validate(text)
  except ValidationError as e:
    return False, e.message

  # Step 2: Extract allowed dia_id from dialogue_text if provided
  allowed = set()
  if dialogue_text:
    allowed = set(DIA_EXTRACT_RE.findall(dialogue_text))

  # Step 3: Validate id-origin consistency and dia_id presence
  for i, s in enumerate(text.get("sentence", [])):
    sid = s.get("id", "")
    origin = s.get("origin", "")

    # 1) Check id format
    if not ID_RE.fullmatch(sid):
      msg = f"sentence[{i}].id format error: {sid}"
      return False, msg

    # 2) Check origin format
    if not ORIGIN_RE.fullmatch(origin):
      msg = f"sentence[{i}].origin format error: {origin}"
      return False, msg

    origin_ids = [x.strip() for x in origin.split(",") if x.strip()]
    if not origin_ids:
      msg = f"sentence[{i}].origin has no source ids: {origin}"
      return False, msg

    # 3) Check first origin == id prefix
    prefix = sid.split("-")[0]
    if origin_ids[0] != prefix:
      msg = f"sentence[{i}]: first origin({origin_ids[0]}) != id prefix({prefix})"
      return False, msg

    # 4) If dialogue_text is provided, check all source dia_ids exist
    if allowed:
      missing = [oid for oid in origin_ids if oid not in allowed]
      if missing:
        msg = f"sentence[{i}]: origin ids not found in allowed dia_id list: {missing}"
        return False, msg

  return True, ""

def check_key_json(text, ref_obj=None, replace=False):
  from jsonschema import Draft202012Validator, ValidationError

  schema = KEY_SCHEMA
  validator = Draft202012Validator(schema)

  # 1) Schema validation for KEY_SCHEMA
  try:
    validator.validate(text)
  except ValidationError as e:
    return False, e.message

  # 2) Extract allowed sentence IDs from ref_obj
  allowed_sentence_ids = set()
  if ref_obj is None:
    msg = "Missing ref_obj for extracting allowed sentence.id."
    return False, msg

  # ref_obj can be a full schema dict or a list of sentences
  if isinstance(ref_obj, dict):
    src_sentences = ref_obj.get("sentence", [])
  elif isinstance(ref_obj, list):
    src_sentences = ref_obj
  else:
    msg = "Unsupported ref_obj type, should be dict (with 'sentence') or list (of sentences)."
    return False, msg

  for s in src_sentences:
    if isinstance(s, dict):
      sid = s.get("id")
      if isinstance(sid, str):
        allowed_sentence_ids.add(sid)

  if not allowed_sentence_ids:
    msg = "No sentence.id extracted from ref_obj."
    return False, msg

  # 3) Validate that KEY_SCHEMA sentence[].sentence_id exists in allowed_sentence_ids
  sentences = text.get("sentence", [])
  for i, s in enumerate(sentences):
    sid = s.get("sentence_id", "")
    if sid not in allowed_sentence_ids:
      if replace:
        text["sentence"] = [
          item for item in sentences
          if isinstance(item, dict) and item.get("sentence_id", "") in allowed_sentence_ids
        ]
        return True, ""
      msg = f"sentence[{i}].sentence_id({sid!r}) not found in allowed sentence.id set"
      return False, msg

  return True, ""
