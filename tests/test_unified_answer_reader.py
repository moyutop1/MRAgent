import inspect
import sys
import types
import unittest


dotenv_module = types.ModuleType("dotenv")
dotenv_module.load_dotenv = lambda: None
sys.modules.setdefault("dotenv", dotenv_module)

from agent.eaes import EAESMixin
from prompts.prompts import Prompts


class UnifiedAnswerReaderTests(unittest.TestCase):
    def test_reader_has_no_category_specific_answer_path(self):
        signature = inspect.signature(EAESMixin._read_eaes_candidates)
        source = inspect.getsource(EAESMixin._read_eaes_candidates)

        self.assertNotIn("category", signature.parameters)
        self.assertNotIn("config.dataset", source)
        self.assertNotIn("TEMPORAL_ANSWER_POLICY", source)
        self.assertNotIn("_eaes_temporal_answer", source)

    def test_temporal_postprocessor_is_removed(self):
        self.assertFalse(hasattr(EAESMixin, "_eaes_temporal_answer"))
        self.assertFalse(hasattr(Prompts, "TEMPORAL_ANSWER_POLICY"))

    def test_final_prompt_is_question_type_agnostic(self):
        prompt = Prompts.EAES_FINAL_ANSWER_PROMPT.lower()

        self.assertNotIn("for time questions", prompt)
        self.assertNotIn("single-time question", prompt)
        self.assertNotIn("exact date", prompt)
        self.assertIn("give the minimal answer requested by the question", prompt)

    def test_public_answer_path_does_not_forward_category_to_reader(self):
        source = inspect.getsource(EAESMixin.answer_question_eaes)

        self.assertIn("lm_current_date=lm_current_date", source)
        reader_call = source.split("self._read_eaes_candidates(", 1)[1]
        reader_call = reader_call.split(")", 1)[0]
        self.assertNotIn("category", reader_call)


if __name__ == "__main__":
    unittest.main()
