"""Actual c1_fuse registration and measured W4 dispatch, without a GPU."""
import importlib
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
import unittest
import tempfile
import subprocess
import shlex
from unittest import mock
import torch
import triton  # native modules must initialize before the isolated fake-vLLM registry
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class AccuracyRegistration(unittest.TestCase):
    def fake_vllm(self):
        names=['vllm','vllm.compilation','vllm.compilation.breakable_cudagraph','vllm.config','vllm.config.compilation',
            'vllm.model_executor','vllm.model_executor.model_loader','vllm.model_executor.model_loader.base_loader',
            'vllm.model_executor.models','vllm.model_executor.models.qwen3_next','vllm.utils','vllm.utils.torch_utils']
        modules={name:ModuleType(name) for name in names}
        for name,module in modules.items():
            if '.' in name:
                parent,child=name.rsplit('.',1);setattr(modules[parent],child,module)
        modules['vllm'].__version__='0.31.0'
        modules['vllm.compilation.breakable_cudagraph'].eager_break_during_capture=lambda fn:fn
        modules['vllm.config.compilation'].CompilationConfig=SimpleNamespace(_attention_ops=[])
        loader=modules['vllm.model_executor.model_loader.base_loader'];loader.process_weights_after_loading=mock.Mock()
        qn=modules['vllm.model_executor.models.qwen3_next']
        qn.Qwen3NextModel=type('Model',(),{'forward':lambda *a:None})
        qn.Qwen3NextDecoderLayer=type('Decoder',(),{})
        ops=[];modules['vllm.utils.torch_utils'].LayerNameType=str
        modules['vllm.utils.torch_utils'].direct_register_custom_op=lambda name,*a,**kw:ops.append(name)
        return modules,ops

    def test_actual_register_accepts_no_w3_and_registers_w4_fusions_without_cuda_build(self):
        modules,ops=self.fake_vllm()
        import atdev_serve
        tree=SimpleNamespace(_register_gated_op=mock.Mock())
        with mock.patch.dict(sys.modules,modules),mock.patch.object(atdev_serve,'tree_attn',tree,create=True),mock.patch.dict(os.environ,{'ATDEV_OPT':'marlin_plus,c1_fuse,dflash_tree,tree_attn','VLLM_MARLIN_INPUT_DTYPE':'bfloat16'}),mock.patch('torch.utils.cpp_extension.load',side_effect=AssertionError('no JIT at registration')):
            fuse=importlib.import_module('atdev_serve.c1_fuse')
            with mock.patch.object(fuse,'_REGISTERED',False):
                error=None
                try:fuse.register()
                except Exception as exc:error=exc
                self.assertIsNone(error,str(error))
        self.assertTrue({'atdev_e6_prod','atdev_e6_cons','atdev_e6_cons_r','atdev_e6_cons_bat','atdev_gdn_core_bap'}<=set(ops))

    def test_w4_fallback_math_ba_dispatch_and_gate_up_permutation(self):
        from atdev_serve import c1_accuracy as acc
        from atdev_serve import c1_norm as cn
        x=torch.arange(12,dtype=torch.float32).reshape(3,4).bfloat16()/10
        weight=torch.arange(24,dtype=torch.float32).reshape(6,4).bfloat16()/100
        res=torch.ones(3,6).bfloat16();nw=torch.arange(6).bfloat16()/10
        def linear(value,wq,ws,n,k):return torch.nn.functional.linear(value,wq)
        with mock.patch.object(acc,'_mp',side_effect=linear),mock.patch.object(acc,'_fused_ok',return_value=False),mock.patch.object(acc,'_tn16',return_value=2):
            a,ro,p=acc.e6_prod(x,weight,torch.ones(1),res,nw,6,4)
            full=torch.nn.functional.linear(x,weight).float()+res.float()
            self.assertTrue(torch.equal(a,(full*(1+nw.float())).bfloat16()))
            self.assertTrue(torch.equal(ro,full.bfloat16()));self.assertTrue(torch.equal(p[:,0],full.square().sum(1)))
            self.assertEqual(torch.count_nonzero(p[:,1:]).item(),0)
            w2=torch.arange(24,dtype=torch.float32).reshape(4,6).bfloat16()/20
            y,r=acc.e6_cons(a,w2,torch.ones(1),p,4,6,6,1e-6,False)
            expected_r=torch.rsqrt(p.sum(1)/6+1e-6)
            self.assertTrue(torch.equal(y,torch.nn.functional.linear((a.float()*expected_r[:,None]).bfloat16(),w2)))
            self.assertTrue(torch.equal(r,expected_r))
        qkv=torch.ones(3,8).bfloat16();r=torch.full((3,),.5);ba=torch.full((3,2),2.).bfloat16()
        with mock.patch.object(acc,'e6_cons_r',return_value=(qkv,r)) as consumer,mock.patch.object(acc.fast,'e6w3_cons_r',side_effect=AssertionError('W3 forbidden')),mock.patch.object(acc.fast,'e6_ba',return_value=ba):
            actual=acc.e6_cons_bat(x,torch.ones(1),torch.ones(1),torch.ones(3,1),8,4,4,1e-6,torch.ones(2,4),False)
            consumer.assert_called_once();self.assertTrue(torch.equal(actual[0],qkv))
            self.assertTrue(torch.equal(actual[1],torch.ones(3)));self.assertTrue(torch.equal(actual[2],ba.float()))
        q=torch.arange(1024,dtype=torch.int32).reshape(2,512);scale=torch.arange(256).reshape(1,256).bfloat16()
        qp,sp=cn.permute_gu_marlin(q,scale,256)
        self.assertEqual(qp[0,128].item(),256);self.assertEqual(qp[0,256].item(),128)
        self.assertEqual(sp[0,64].item(),128);self.assertEqual(sp[0,128].item(),64)
        self.assertEqual(q[0,128].item(),128)

    def test_real_cli_help_runs_full_accuracy_registry_with_local_draft_and_hardware_only_stubs(self):
        root=Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);draft=folder/'draft';draft.mkdir();(draft/'config.json').write_text('{}')
            wrapper=folder/'vllm'
            wrapper.write_text('#!/bin/sh\nexec '+shlex.quote(sys.executable)+' '+shlex.quote(str(root/'tests/cpu_registry_help.py'))+' "$@"\n');wrapper.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith(('ATDEV_','VLLM_','CUDA_'))}
            env.update(PYTHON=sys.executable,PYTHONPATH=str(root/'src'),VLLM=str(wrapper),VLLM_PLUGINS='atdev',HF_HUB_OFFLINE='1',DRAFTER=str(draft),CUDA_VISIBLE_DEVICES='')
            result=subprocess.run(['bash',str(root/'serve_c1_accuracy.sh'),'--help'],cwd=root,env=env,text=True,capture_output=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stderr[-10000:])
            self.assertIn('usage: vllm serve',result.stdout)
            self.assertIn('measured W4 norm/SiLU fusions registered',result.stderr)
            self.assertNotIn('w3_mixed',result.stderr)

if __name__=='__main__':unittest.main()
