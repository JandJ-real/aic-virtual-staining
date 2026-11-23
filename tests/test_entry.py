import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

class EntryTests(unittest.TestCase):
 def test_offline_cli_exposes_check_infer_and_train(self):
  r=subprocess.run([sys.executable,str(ROOT/'reproduce.py'),'--help'],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)
  for command in ('check','infer','train'):self.assertIn(command,r.stdout)

 def contract(self):return importlib.import_module('repro_contract')

 def test_runtime_distribution_local_suffix_is_separate_from_cuda_build(self):
  c=self.contract();v=dict(python='3.9.18',torch='2.4.1',numpy='1.20.3',Pillow='8.4.0',scipy='1.7.1',**{'scikit-image':'0.18.3'})
  self.assertTrue(c.original_runtime(v,'2.4.1+cu121','12.1',90100))
  v['torch']='2.4.1+cu121'
  self.assertTrue(c.original_runtime(v,'2.4.1+cu121','12.1',90100))

 def test_runtime_rejects_cpu_build_wrong_cuda_and_dependency(self):
  c=self.contract();v=dict(python='3.9.18',torch='2.4.1',numpy='1.20.3',Pillow='8.4.0',scipy='1.7.1',**{'scikit-image':'0.18.3'})
  self.assertFalse(c.original_runtime(v,'2.4.1+cpu','12.1',90100))
  self.assertFalse(c.original_runtime(v,'2.4.1+cu121','11.8',90100))
  self.assertFalse(c.original_runtime(dict(v,numpy='2.0.0'),'2.4.1+cu121','12.1',90100))

 def test_main_learning_rate_at_all_phase_boundaries(self):
  c=self.contract()
  for step,lr in [(1,1e-5),(250,1e-5),(251,5e-5),(500,5e-5),(501,1e-4),(11200,1e-4),(11201,5e-5),(14000,5e-5)]:
   self.assertEqual(c.main_lr(step),lr)
  for step in (0,14001):
   with self.assertRaises(ValueError):c.main_lr(step)

 def test_only_exact_final_avg_checkpoint_is_accepted(self):
  c=self.contract();summary=json.loads((ROOT/'tests/avg_fixture.json').read_text())
  p={'contract':summary['contract'],'tail_updates':2450,'avg_sources':summary['sources']}
  self.assertTrue(c.validate_endpoint(p,'042816bbab9fd844738f68348f12b761ba0e87c0ce703de3b541d63a83e88922'))
  for mutation in ('digest','partial','anchor','marker','val','init'):
   bad=copy.deepcopy(p);digest=c.FINAL_SHA
   if mutation=='digest':digest='a'*64
   elif mutation=='partial':bad['tail_updates']=2449
   elif mutation=='anchor':bad['contract']['anchor_step']=12000
   elif mutation=='marker':bad['contract']['source_contract']['task_markers']['liver']='CD68'
   elif mutation=='val':bad['contract']['source_contract']['independent_validation']='A'
   else:bad['contract']['source_contract']['initialization']='SSL150'
   with self.subTest(mutation=mutation),self.assertRaises(ValueError):c.validate_endpoint(bad,digest)

if __name__=='__main__':unittest.main()
