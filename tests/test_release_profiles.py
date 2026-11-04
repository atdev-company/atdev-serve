"""CPU-only public launcher parity; fake vLLM records allowlisted env/argv."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
FAST='gdn_single_state,dflash_tree,attn_split_verify,gdn_meta_fast,shadow_head,draft_vocab,marlin_plus,ba_gemv,c1_fuse,drafter_ctxfold,w3_mixed,verify_split_head,drafter_w4,tree_attn,ll1_fuse,ll5_fuse'
C32='a4_sim,draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse'

class Profiles(unittest.TestCase):
    def launch(self,script,**overrides):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'vllm'
            fake.write_text('#!'+sys.executable+'\nimport os,sys,json\nprint(json.dumps({"argv":sys.argv[1:],"env":{k:v for k,v in os.environ.items() if k.startswith(("ATDEV_","VLLM_","CUDA_"))}}))\n');fake.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith(('ATDEV_','VLLM_','CUDA_','C1_MODE','C32_HADAMARD','HADW'))}
            env.update(PYTHON=sys.executable,PYTHONPATH=str(ROOT/'src'),VLLM=str(fake),**overrides)
            run=subprocess.run(['bash',str(ROOT/script),'--seed','17'],env=env,text=True,capture_output=True,cwd=ROOT)
            return run,json.loads(run.stdout) if run.returncode==0 else None

    def test_c1_default_fast_exact_features_and_dflash_args(self):
        run,p=self.launch('serve_c1.sh');self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(p['env']['ATDEV_OPT'],FAST)
        self.assertEqual(p['env']['ATDEV_W3_SIDECAR'],'atdev-company/Qwen3.8-27B-atdev-c1:w3/')
        self.assertEqual(p['env']['ATDEV_DRAFTER_QUANT_PATH'],'atdev-company/Qwen3.8-27B-atdev-c1:drafter/')
        self.assertEqual(json.loads(p['argv'][p['argv'].index('--speculative-config')+1]),{'method':'dflash','model':'z-lab/Qwen3.8-27B-DFlash2','num_speculative_tokens':15})
        self.assertEqual(p['argv'][-2:],['--seed','17']);self.assertIn('c1-fast',run.stderr)

    def test_accuracy_changes_only_w3_feature_and_sidecar_not_other_flags_or_argv(self):
        _,fast=self.launch('serve_c1.sh')
        run,acc=self.launch('serve_c1.sh',C1_MODE='accuracy',ATDEV_W3_SIDECAR='must-not-load')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(acc['argv'],fast['argv'])
        expected=dict(fast['env']);expected['ATDEV_OPT']=','.join(x for x in FAST.split(',') if x!='w3_mixed');expected.pop('ATDEV_W3_SIDECAR')
        self.assertEqual(acc['env'],expected);self.assertIn('c1-accuracy',run.stderr)
        wrapper,p=self.launch('serve_c1_accuracy.sh');self.assertEqual(wrapper.returncode,0,wrapper.stderr);self.assertEqual(p,acc)
        invalid,_=self.launch('serve_c1.sh',C1_MODE='typo');self.assertNotEqual(invalid.returncode,0)

    def test_c32_only_selected_hadamard_with_builtin_mtp(self):
        run,p=self.launch('serve_c32.sh',ATDEV_A4_REAL='unrotated',ATDEV_DRAFTER_QUANT_PATH='must-not-load')
        self.assertEqual(run.returncode,0,run.stderr);e=p['env'];self.assertEqual(e['ATDEV_OPT'],C32)
        self.assertEqual(e['ATDEV_A4_REAL'],'had3')
        self.assertEqual(e['ATDEV_A4_HADW'],'atdev-company/Qwen3.8-27B-atdev-c32:hadamard/qwen38_27b_hadamard_gptq_d10.safetensors')
        self.assertTrue(e['ATDEV_GDN_SRC'].endswith('/csrc/gdn_var/gdn_ss12h.cu'))
        self.assertNotIn('ATDEV_DRAFTER_QUANT_PATH',e)
        selected={'ATDEV_FQ_A4_NORM':'16,8','ATDEV_FQ_A4_SILU':'16,4','ATDEV_C32_MK':'1',
            'ATDEV_C32K_SCR':'1','ATDEV_C32C_CSUM':'1','ATDEV_GDN_CORE_EMPTY':'1','ATDEV_C32_QKG':'1',
            'ATDEV_FQ_A4_CUDA':'1','ATDEV_DI1_CG_MAXREQS':'32','ATDEV_DI1_BUDGET':'rows:96',
            'ATDEV_VSPLIT_A4SCR':'1','ATDEV_VSPLIT_A4SCR_CFG':'16,4','ATDEV_VSPLIT_TAIL_MP':'64,256,256,6,1,4',
            'ATDEV_A4_DEFINES':'A4_FOLD2=1,A4_MAGICQ=1,A4_SCALAR_CTMP=1,A4_PAR_REDUCE=6,A4_C32C_PB=1',
            'VLLM_MARLIN_INPUT_DTYPE':'int8'}
        for key,value in selected.items():self.assertEqual(e[key],value,key)
        self.assertEqual(json.loads(p['argv'][p['argv'].index('--speculative-config')+1]),{'method':'mtp','num_speculative_tokens':3,'use_local_argmax_reduction':True})
        self.assertIn('c32-hadamard-w4a4',run.stderr)
        _,custom=self.launch('serve_c32.sh',C32_HADAMARD='/tmp/custom.safetensors');self.assertEqual(custom['env']['ATDEV_A4_HADW'],'/tmp/custom.safetensors')

    def test_accuracy_registry_never_imports_w3_or_resolves_overlay(self):
        sys.path.insert(0,str(ROOT/'src'))
        from types import SimpleNamespace
        from unittest import mock
        import atdev_serve
        from atdev_serve import register, hub
        called=[]
        def imported(name):
            called.append(name)
            return SimpleNamespace(__file__='fake_module.py',register=lambda:None)
        cache=SimpleNamespace(_SALTS={},salt_aot_cache=lambda *a:None)
        pin=SimpleNamespace(register=lambda:None,register_inductor=lambda:None)
        accuracy=','.join(x for x in FAST.split(',') if x!='w3_mixed')
        with mock.patch.object(register,'_DONE',False),mock.patch.object(atdev_serve,'compile_cache',cache,create=True),mock.patch.object(atdev_serve,'pin_autotune',pin,create=True),mock.patch.dict(sys.modules,{'atdev_serve.compile_cache':cache,'atdev_serve.pin_autotune':pin}),mock.patch.object(register.importlib,'import_module',side_effect=imported),mock.patch.dict(os.environ,{'ATDEV_OPT':accuracy}),mock.patch.object(hub,'resolve_files',side_effect=AssertionError('overlay resolution forbidden')):
            register.register_all()
        self.assertNotIn('atdev_serve.w3_mixed',called)
        self.assertIn('atdev_serve.drafter_w4',called)

if __name__=='__main__':unittest.main()
