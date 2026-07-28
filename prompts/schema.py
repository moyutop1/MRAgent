
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
        "required": ["id", "text", "tag", "origin", "topic", "semantic_properties"],
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
          "semantic_properties": {
            "type": "array",
            "uniqueItems": True,
            "items": {
              "type": "string",
              "enum": [
                "event_action",
                "state_opinion",
                "personal_profile",
                "relation_social",
                "transient",
                "episodic",
                "durable",
                "unknown"
              ]
            }
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
            "pattern": "^D\\d+:\\d+(?:-\\d+)?$",
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
from copy import deepcopy
from typing import List, Dict, Any, Tuple, Set

ID_RE = re.compile(r'^D\d+:\d+-\d+$')
ORIGIN_RE = re.compile(r'^D\d+:\d+(,\s*D\d+:\d+)*$')
DIA_EXTRACT_RE = re.compile(r'dia_id\s*:\s*(D\d+:\d+)', re.IGNORECASE)


def check_rewrite_json(text, dialogue_text, allow_origin_id=False):
  from jsonschema import Draft202012Validator, ValidationError
  import re
  schema = deepcopy(SCHEMA) if allow_origin_id else SCHEMA
  if allow_origin_id:
    schema["properties"]["sentence"]["items"]["properties"]["id"]["pattern"] = (
      "^D\\d+:\\d+(?:-\\d+)?$"
    )

  validator = Draft202012Validator(schema)
  id_pattern = r'^D\d+:\d+(?:-\d+)?$' if allow_origin_id else r'^D\d+:\d+-\d+$'
  ID_RE = re.compile(id_pattern)
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
    semantic_properties = s.get("semantic_properties")

    # The semantic field combines two independent axes: up to three content
    # properties plus exactly one persistence property. Schema validation above
    # already enforces list type, uniqueness, and the complete label whitelist.
    content_properties = {
      "event_action", "state_opinion", "personal_profile", "relation_social"
    }
    persistence_properties = {"transient", "episodic", "durable", "unknown"}
    content_count = sum(item in content_properties for item in semantic_properties)
    persistence_count = sum(item in persistence_properties for item in semantic_properties)
    if content_count > 3:
      return False, f"sentence[{i}].semantic_properties has more than 3 content properties"
    if persistence_count != 1:
      return False, (
        f"sentence[{i}].semantic_properties must contain exactly one "
        "persistence property"
      )

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


def check_child_batch_rewrite_json(text, child_segments, dialogue_text):
  """Validate the strict one-child-segment to one-rewrite-sentence contract."""
  flag, err = check_rewrite_json(text, dialogue_text, allow_origin_id=True)
  if not flag:
    return flag, err
  if not isinstance(child_segments, list) or len(child_segments) > 15:
    return False, "child batch must be a list containing at most 15 segments"
  sentences = text.get("sentence") or []
  if len(sentences) != len(child_segments):
    return False, (
      f"child batch cardinality mismatch: expected {len(child_segments)} "
      f"sentences, got {len(sentences)}"
    )
  if text.get("topics") not in ({}, None):
    return False, "hierarchical child rewrite topics must be empty"
  for index, (sentence, child) in enumerate(zip(sentences, child_segments)):
    child_id = getattr(child, "child_id", None)
    current_turns = getattr(child, "current_turns", [])
    if sentence.get("id") != child_id:
      return False, (
        f"sentence[{index}].id must echo {child_id!r}, "
        f"got {sentence.get('id')!r}"
      )
    if sentence.get("topic") != []:
      return False, f"sentence[{index}].topic must be []"
    current_ids = {getattr(turn, "origin", None) for turn in current_turns}
    used_order = re.findall(r"D\d+:\d+", sentence.get("origin") or "")
    used_ids = set(used_order)
    if not used_ids or not used_ids.issubset(current_ids):
      return False, f"sentence[{index}] uses origins outside its child window"
    source_origins = getattr(child, "source_origins", [])
    if not source_origins or used_order[0] != source_origins[0]:
      return False, (
        f"sentence[{index}].origin must start with the child window's first "
        f"source origin {source_origins[0] if source_origins else None!r}"
      )
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
