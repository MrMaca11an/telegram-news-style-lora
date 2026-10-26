import unittest
from src.common import ROOT,read_jsonl,output_checks,prompt_messages
from src.prepare import prepare,digest
from src.build_synthetic import main as build_synth

class LabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        build_synth()
        prepare(synthetic_count=0,variant='orig')
        prepare(synthetic_count=10,variant='mix10')
        prepare(synthetic_count=20,variant='mix20')
    def test_counts(self):
        for style in ['type1','type2']:
            self.assertEqual(len(read_jsonl(ROOT/f'data/raw/train_{style}.jsonl')),25)
            self.assertEqual(len(read_jsonl(ROOT/f'data/processed/orig/{style}_train.jsonl')),20)
            self.assertEqual(len(read_jsonl(ROOT/f'data/processed/mix10/{style}_train.jsonl')),30)
            self.assertEqual(len(read_jsonl(ROOT/f'data/processed/mix20/{style}_train.jsonl')),40)
            self.assertEqual(len(read_jsonl(ROOT/f'data/processed/orig/{style}_val.jsonl')),5)
    def test_val_constant(self):
        for style in ['type1','type2']:
            vals=[[x['id'] for x in read_jsonl(ROOT/f'data/processed/{v}/{style}_val_manifest.jsonl')]
                  for v in ['orig','mix10','mix20']]
            self.assertEqual(vals[0],vals[1]);self.assertEqual(vals[1],vals[2])
    def test_no_exact_leakage(self):
        tests={digest(x['text']) for x in read_jsonl(ROOT/'data/raw/test_inputs.jsonl')}
        for style in ['type1','type2']:
            for v in ['orig','mix10','mix20']:
                train={digest(x['input']) for x in read_jsonl(ROOT/f'data/processed/{v}/{style}_train_manifest.jsonl')}
                val={digest(x['input']) for x in read_jsonl(ROOT/f'data/processed/{v}/{style}_val_manifest.jsonl')}
                self.assertFalse(train&val);self.assertFalse(train&tests)
    def test_synth_format(self):
        for style in ['type1','type2']:
            rows=read_jsonl(ROOT/f'data/synthetic/synthetic_{style}.jsonl')
            self.assertEqual(len(rows),20)
            for r in rows:self.assertTrue(output_checks(style,r['output'])['pass'],r['id'])
    def test_prompt(self):
        self.assertEqual(prompt_messages('type1','Тест')[0]['role'],'system')
        self.assertIn('@banki_oil',prompt_messages('type1','Тест')[0]['content'])
    def test_test_count(self):
        self.assertEqual(len(read_jsonl(ROOT/'data/raw/test_inputs.jsonl')),5)

if __name__ == '__main__':unittest.main()
