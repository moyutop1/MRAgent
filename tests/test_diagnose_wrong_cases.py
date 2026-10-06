import unittest

from eval.diagnose_wrong_cases import (
    add_case,
    build_case,
    case_key,
    find_case,
    is_hit20,
)


class DiagnoseWrongCasesTest(unittest.TestCase):
    def test_case_key_prefers_question_index_over_formatted_question(self):
        answer_row = {
            "sample": "conv-42",
            "question_index": 7,
            "question": "Where did they meet? No extra explanations. ",
        }
        retrieval_row = {
            "sample": "conv-42",
            "question_index": 7,
            "question": "Where did they meet?",
        }

        self.assertEqual(case_key(answer_row), case_key(retrieval_row))

    def test_case_key_normalizes_string_question_index(self):
        self.assertEqual(
            case_key({"sample": "conv-42", "question_index": 7}),
            case_key({"sample": "conv-42", "question_index": " 7 "}),
        )

    def test_case_key_strips_category_prompt_when_index_is_missing(self):
        self.assertEqual(
            case_key({"sample": "conv-42", "question": "Where did they meet? No extra explanations. "}),
            case_key({"sample": "conv-42", "question": "Where did they meet?"}),
        )

    def test_lookup_falls_back_to_question_when_only_one_row_has_index(self):
        indexed = {
            "sample": "conv-42",
            "question_index": 7,
            "question": "Where did they meet?",
        }
        answer = {
            "sample": "conv-42",
            "question": "Where did they meet? No extra explanations. ",
        }
        by_key = {}
        add_case(by_key, indexed)

        self.assertIs(find_case(by_key, answer), indexed)

    def test_is_hit20_accepts_json_boolean_and_integer(self):
        self.assertTrue(is_hit20({"hit_at_20": True}))
        self.assertTrue(is_hit20({"hit_at_20": 1}))
        self.assertFalse(is_hit20({"hit_at_20": False}))
        self.assertFalse(is_hit20(None))

    def test_build_case_preserves_answer_and_retrieval_errors(self):
        case = build_case(
            {
                "sample": "conv-26",
                "question_index": 2,
                "prediction": "ERROR",
                "error_type": "APIStatusError",
                "error": "Error code: 400",
            },
            None,
            {"error": "phrase generation failed", "retrieval": {}},
            8,
        )

        self.assertEqual(case["answer_error_type"], "APIStatusError")
        self.assertEqual(case["answer_error"], "Error code: 400")
        self.assertEqual(case["retrieval_error"], "phrase generation failed")


if __name__ == "__main__":
    unittest.main()
