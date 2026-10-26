"""CPU-only tests: chat-template label masking invariants, length accounting,
quality_eval heuristics, deterministic postprocess. No GPU, no model weights."""
import unittest
from src.common import ROOT, read_jsonl
from src.quality_eval import structural_checks, numbers_normalized, numeric_audit
from src.postprocess import format_post

TOKENIZER_CANDIDATES = [ROOT / 'artifacts' / 'type1_mix10', ROOT / 'adapters' / 'type1']
TOKENIZER_DIR = next((p for p in TOKENIZER_CANDIDATES if p.exists()), TOKENIZER_CANDIDATES[0])
EMOJI = chr(0x1F4F0)


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))


@unittest.skipUnless(TOKENIZER_DIR.exists(), 'tokenizer artifact not available')
class TestLabelMasking(unittest.TestCase):
    def test_prompt_is_strict_prefix_of_full(self):
        tok = _tok()
        row = read_jsonl(ROOT / 'data/processed/orig/type1_train.jsonl')[0]
        prompt_ids = tok.apply_chat_template(row['prompt'], tokenize=True,
            add_generation_prompt=True, enable_thinking=False)
        full_ids = tok.apply_chat_template(row['prompt'] + row['completion'], tokenize=True,
            add_generation_prompt=False, enable_thinking=False)
        if hasattr(full_ids, 'input_ids'):
            full_ids = full_ids['input_ids']
        if hasattr(prompt_ids, 'input_ids'):
            prompt_ids = prompt_ids['input_ids']
        self.assertIsInstance(prompt_ids, list, 'expected flat token id list')
        self.assertGreater(len(prompt_ids), 10, 'length must be real token count, not dict len')
        self.assertEqual(list(full_ids[:len(prompt_ids)]), list(prompt_ids)[:len(prompt_ids)])
        self.assertGreater(len(full_ids), len(prompt_ids))

    def test_completion_text_not_in_prompt_span(self):
        tok = _tok()
        row = read_jsonl(ROOT / 'data/processed/orig/type2_train.jsonl')[0]
        completion_text = row['completion'][-1]['content']
        head = completion_text[:12]
        prompt_only = tok.apply_chat_template(row['prompt'], tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        self.assertNotIn(head, prompt_only)
        full = tok.apply_chat_template(row['prompt'] + row['completion'], tokenize=False,
            add_generation_prompt=False, enable_thinking=False)
        self.assertIn(head, full)


class TestQualityEval(unittest.TestCase):
    def test_type1_signature_inline(self):
        ok = 'Рынок вырос. Об этом заявили аналитики. @banki_oil'
        self.assertTrue(structural_checks('type1', ok)['pass'])
        bad = 'Рынок вырос.\n@banki_oil'
        self.assertFalse(structural_checks('type1', bad)['pass'])

    def test_emoji_grapheme_start(self):
        ok = EMOJI + ' Тест новость.\n\nДетали происшествия.'
        self.assertTrue(structural_checks('type2', ok)['starts_emoji_grapheme'])
        self.assertTrue(structural_checks('type2', ok)['pass'])
        bad = 'Тест новость.\n\nДетали происшествия.'
        self.assertFalse(structural_checks('type2', bad)['starts_emoji_grapheme'])

    def test_three_paragraphs_fail(self):
        bad = EMOJI + ' Лид.\n\nВторой абзац.\n\nТретий абзац.'
        self.assertFalse(structural_checks('type2', bad)['pass'])

    def test_numbers_normalized(self):
        self.assertEqual(numbers_normalized('1,5-2 раза и 50%'), {1.5, 2.0, 50})
        self.assertEqual(numbers_normalized('до 30 тысяч'), {30})
        self.assertEqual(numbers_normalized('11-17 м²'), {11, 17})

    def test_numeric_audit_flags_missing(self):
        a = numeric_audit('штраф до 30 тысяч рублей', 'штраф увеличен')
        self.assertEqual(a['missing_critical'], [30])


class TestPostprocess(unittest.TestCase):
    def test_type1_markdown_and_signature(self):
        s = format_post('type1', '**Рынок** вырос\nна 2%.')
        self.assertNotIn('**', s)
        self.assertTrue(s.endswith('@banki_oil'))
        self.assertNotIn('\n', s)

    def test_type2_merges_extra_paragraph(self):
        s = format_post('type2', EMOJI + ' Лид.\n\nВторой абзац.\n\nТретий.')
        self.assertEqual(len([p for p in s.split('\n\n') if p.strip()]), 2)


if __name__ == '__main__':
    unittest.main()
