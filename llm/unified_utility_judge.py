import json
import threading
from pathlib import Path

from common import config


_JUDGE = None
_JUDGE_LOCK = threading.Lock()


class UnifiedUtilityJudge:
    """One local instruction model for WRITE, ABSTRACT, and SELECT gain."""

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        model_path = Path(config.UTILITY_MODEL_PATH).expanduser().resolve()
        if not model_path.is_dir():
            raise ValueError(
                f"UTILITY_MODEL_PATH must be a local model directory: {model_path}"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if config.UTILITY_DEVICE != "cpu" else torch.float32,
            device_map="auto" if config.UTILITY_DEVICE != "cpu" else None,
            local_files_only=True,
        )
        if config.UTILITY_DEVICE == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self._generation_lock = threading.Lock()

    def count_tokens(self, value):
        text = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def keep(self, operation, condition, selected_set, candidate, source=None):
        payload = {
            "operation": operation,
            "condition": condition,
            "selected_set": selected_set,
            "candidate": candidate,
        }
        if source is not None:
            payload["source"] = source
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a unified set-conditioned marginal-utility judge. "
                    "Decide whether CANDIDATE adds useful information under CONDITION "
                    "given SELECTED_SET. The same rule applies to every operation: KEEP "
                    "only when the candidate has positive conditional marginal value.\n"
                    "WRITE: KEEP only if it adds a new event/state, time/stage, or concrete "
                    "detail; DROP only when it describes the same event/state at the same "
                    "time/stage and adds no detail.\n"
                    "ABSTRACT: KEEP only if the abstraction is supported by SOURCE, is "
                    "more useful than merely repeating its children, and adds a distinct "
                    "profile, event synopsis, entity relation, or stable high-level state.\n"
                    "SELECT: KEEP only if it adds information useful for answering the query "
                    "that is not already supplied by SELECTED_SET.\n"
                    "Return exactly KEEP or DROP."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if inputs.input_ids.shape[1] > config.UTILITY_MAX_LENGTH:
            raise ValueError(
                f"utility input has {inputs.input_ids.shape[1]} tokens; "
                f"maximum is {config.UTILITY_MAX_LENGTH}"
            )
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        with self._generation_lock:
            output = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        answer = self.tokenizer.decode(
            output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip().upper()
        return answer.startswith("KEEP")


def get_unified_utility_judge():
    global _JUDGE
    if _JUDGE is None:
        with _JUDGE_LOCK:
            if _JUDGE is None:
                _JUDGE = UnifiedUtilityJudge()
    return _JUDGE
