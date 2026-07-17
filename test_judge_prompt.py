import ast
from pathlib import Path
import unittest


def _load_accuracy_prompt():
    source_path = Path(__file__).parent / "eval" / "judge.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    for node in module.body:
        if (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id == "ACCURACY_PROMPT"
                    for target in node.targets
                )
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("ACCURACY_PROMPT was not found")


class JudgePromptTests(unittest.TestCase):
    def test_prompt_accepts_semantically_equivalent_paraphrases(self):
        prompt = _load_accuracy_prompt()

        self.assertIn("Judge semantic equivalence, not lexical overlap", prompt)
        self.assertIn('Gold: "school speech"', prompt)
        self.assertIn('Generated: "gave a talk at a school event" -> CORRECT', prompt)

    def test_prompt_rejects_answers_that_only_share_a_topic(self):
        prompt = _load_accuracy_prompt()

        self.assertIn("sharing only a broad topic is not enough", prompt)
        self.assertIn('Generated: "attended a school event" -> WRONG', prompt)
        self.assertIn("planned versus completed", prompt)

    def test_prompt_accepts_complete_gold_with_additional_answers(self):
        prompt = _load_accuracy_prompt()

        self.assertIn("Use a coverage-first rule", prompt)
        self.assertIn(
            'Gold: "painting and pottery"; Generated: "painting, pottery, and photography" -> CORRECT',
            prompt,
        )
        self.assertIn("do not mark it WRONG merely because additional answers are present", prompt)
        self.assertIn("Merely adding another possible answer is not a contradiction", prompt)


if __name__ == "__main__":
    unittest.main()
