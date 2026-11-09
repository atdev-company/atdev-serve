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
MT_C1={'ATDEV_DFT_APC':'1','ATDEV_PF_FUSE':'1','ATDEV_PF_GLUE':'1','ATDEV_PF_FA2':'1','ATDEV_PF_FA2_GATHER':'1','ATDEV_PF_FA2_FI':'1','ATDEV_PF_FA2_FI_MINQ':'2','ATDEV_PF_FA2_FI_PINRING':'1','ATDEV_SWA_FINE':'1','ATDEV_W3_DQ':'1','ATDEV_W3_DQ_MIN_M':'1536','ATDEV_W3_I8':'1','ATDEV_W3_I8_MIN_M':'64','ATDEV_TOK_CACHE':'1','ATDEV_GDN_SMALL':'32','ATDEV_TREE_HSPL':'auto','VLLM_USE_V2_MODEL_RUNNER':'1','ATDEV_VSPLIT_FBITS':'8','ATDEV_VSPLIT_STREAMS':'1','ATDEV_DRAFT_VOCAB_TOPKX':'1'}
C32_MT='a4_sim,draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse'
MT_C32={'ATDEV_A4_REAL':'had3','ATDEV_A4ACC_FUSE':'1','ATDEV_A4ACC_S4':'1','ATDEV_A4ACC_CLIP':'0.9','ATDEV_PF_A4_SKIP':'*:0-0','ATDEV_GDN_META':'1','ATDEV_GDN_SS_ALIGN':'1','ATDEV_GDN_IMPL':'cuda','ATDEV_DRAFT_VOCAB_SIZE':'65536','ATDEV_DRAFT_VOCAB_BITS':'4','ATDEV_VSPLIT_SIZE':'65536','ATDEV_VSPLIT_BITS':'4','ATDEV_C32_MK':'1','ATDEV_C32K_SCR':'1','ATDEV_C32C_CSUM':'1','ATDEV_C32P_BA':'1','ATDEV_GDN_CORE_EMPTY':'1','ATDEV_C32_QKG':'1','ATDEV_FQ_A4_CUDA':'1','ATDEV_A4_SET':'cfg','ATDEV_A4_DEFINES':'A4_FOLD2=1,A4_MAGICQ=1,A4_SCALAR_CTMP=1,A4_PAR_REDUCE=6,A4_C32C_PB=1,A4_S0=1,A4_PTRINC=1,A4_PTRINC_ST=6','ATDEV_DI1_CG_MAXREQS':'32','ATDEV_DI1_BUDGET':'rows:96','ATDEV_DI1_STATS':'0','ATDEV_VSPLIT_FAST':'1','ATDEV_VSPLIT_TAIL_MP':'64,256,256,6,1,4','ATDEV_VSPLIT_A4SCR':'1','ATDEV_VSPLIT_A4SCR_CFG':'16,4','ATDEV_PF_FA2':'1','ATDEV_PF_FI':'1','ATDEV_PF_MP':'1','ATDEV_PF_A4':'1','ATDEV_PF_A4_RQV':'1','ATDEV_TOK_CACHE':'1','ATDEV_DRAFT_VOCAB_FUSED':'1','ATDEV_VSPLIT_A8':'1','VLLM_MARLIN_INPUT_DTYPE':'int8'}
C32='a4_sim,draft_vocab,gdn_single_state,verify_split_head,mtp_w4,marlin_plus,fused_quant,vsplit_mtp,gdn_a4_fold,mtp_prune,c32_attn,di1_varlen,c32_conv,c32k_fuse'

class Profiles(unittest.TestCase):
    def launch(self,script,*args,**overrides):
        with tempfile.TemporaryDirectory() as d:
            fake=Path(d)/'vllm'
            fake.write_text('#!'+sys.executable+'\nimport os,sys,json\nprint(json.dumps({"argv":sys.argv[1:],"env":{k:v for k,v in os.environ.items() if k.startswith(("ATDEV_","VLLM_","CUDA_"))}}))\n');fake.chmod(0o755)
            env={k:v for k,v in os.environ.items() if not k.startswith(('ATDEV_','VLLM_','CUDA_','C1_MODE','C32_HADAMARD','HADW'))}
            env.update(PYTHON=sys.executable,PYTHONPATH=str(ROOT/'src'),VLLM=str(fake),**overrides)
            run=subprocess.run(['bash',str(ROOT/script),*args,'--seed','17'],env=env,text=True,capture_output=True,cwd=ROOT)
            return run,json.loads(run.stdout) if run.returncode==0 else None

    def test_c1_single_profile_fast_exact_features_and_dflash_args(self):
        run,p=self.launch('serve_c1.sh','--profile','single');self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(p['env']['ATDEV_OPT'],FAST)
        self.assertEqual(p['env']['ATDEV_W3_SIDECAR'],'atdev-company/Qwen3.8-27B-atdev-c1:w3/')
        self.assertEqual(p['env']['ATDEV_DRAFTER_QUANT_PATH'],'atdev-company/Qwen3.8-27B-atdev-c1:drafter/')
        self.assertEqual(json.loads(p['argv'][p['argv'].index('--speculative-config')+1]),{'method':'dflash','model':'z-lab/Qwen3.8-27B-DFlash2','num_speculative_tokens':15})
        self.assertIn('--no-enable-prefix-caching',p['argv']);self.assertEqual(p['argv'][p['argv'].index('--max-model-len')+1],'8192')
        self.assertNotIn('ATDEV_DFT_APC',p['env'])
        self.assertEqual(p['argv'][-2:],['--seed','17']);self.assertIn('c1-fast',run.stderr)

    def test_c1_default_is_multiturn_with_the_frozen_bundle(self):
        run,p=self.launch('serve_c1.sh','--profile','multiturn',NO_WARMUP='1');default,d=self.launch('serve_c1.sh',NO_WARMUP='1')
        self.assertEqual(run.returncode,0,run.stderr);self.assertEqual(default.returncode,0,default.stderr);self.assertEqual(p,d)
        self.assertEqual(p['env']['ATDEV_OPT'],FAST)
        for key,value in MT_C1.items():self.assertEqual(p['env'][key],value,key)
        self.assertNotIn('ATDEV_EOR_CKPT',p['env'])
        spec=json.loads(p['argv'][p['argv'].index('--speculative-config')+1])
        self.assertEqual(spec,{'method':'dflash','model':'z-lab/Qwen3.8-27B-DFlash2','num_speculative_tokens':15,'disable_eagle_block_drop':True})
        a=p['argv'];self.assertIn('--enable-prefix-caching',a);self.assertEqual(a[a.index('--mamba-cache-mode')+1],'align')
        self.assertIn('--enable-mamba-shared-prefix-checkpoint',a);self.assertEqual(a[a.index('--prefix-match-unit')+1],'16')
        self.assertEqual(a[a.index('--max-model-len')+1],'131072');self.assertEqual(a[a.index('--max-num-batched-tokens')+1],'2048')
        self.assertEqual(a[-2:],['--seed','17']);self.assertIn('multiturn',run.stderr)
        eor,e=self.launch('serve_c1.sh',NO_WARMUP='1',ATDEV_EOR_CKPT='1');self.assertEqual(eor.returncode,0,eor.stderr)
        self.assertEqual((e['env']['ATDEV_EOR_CKPT'],e['env']['ATDEV_EOR_SNAP']),('1','16'))

    def test_c32_multiturn_profile_is_the_frozen_c32mt_bundle_and_single_stays_hadamard(self):
        run,p=self.launch('serve_c32.sh','--profile','multiturn',NO_WARMUP='1');dflt,d=self.launch('serve_c32.sh',NO_WARMUP='1');self.assertEqual(p,d);self.assertEqual(run.returncode,0,run.stderr);e=p['env']
        self.assertEqual(e['ATDEV_OPT'],C32_MT);self.assertTrue(e['ATDEV_A4_HADW'].endswith('qwen38_27b_hadamard_gptq_d10.safetensors'))
        for key,value in MT_C32.items():self.assertEqual(e[key],value,key)
        self.assertTrue(e['ATDEV_GDN_SRC'].endswith('/csrc/gdn_var/gdn_ss15h.cu'));self.assertNotIn('ATDEV_EOR_CKPT',e)
        a=p['argv'];self.assertEqual(a[a.index('--max-model-len')+1],'16384');self.assertEqual(a[a.index('--max-num-batched-tokens')+1],'4096')
        self.assertEqual(a[a.index('--max-num-seqs')+1],'32');self.assertEqual(a[a.index('--gpu-memory-utilization')+1],'0.90')
        self.assertEqual(a[a.index('--prefix-match-unit')+1],'16');self.assertIn('--enable-prefix-caching',a)
        self.assertEqual(json.loads(a[a.index('--speculative-config')+1]),{'method':'mtp','num_speculative_tokens':3,'use_local_argmax_reduction':True})
        _,noa4=self.launch('serve_c32.sh','--profile','multiturn',NO_WARMUP='1',ATDEV_PF_A4='0');self.assertEqual(noa4['env']['ATDEV_PF_A4'],'0')
        eor,x=self.launch('serve_c32.sh','--profile','multiturn',NO_WARMUP='1',ATDEV_EOR_CKPT='1');self.assertEqual(eor.returncode,0,eor.stderr)
        self.assertEqual((x['env']['ATDEV_EOR_CKPT'],x['env']['ATDEV_EOR_SNAP'],x['env']['ATDEV_EOR_PLANNED']),('1','32','1'))
        self.assertEqual(x['argv'][x['argv'].index('--gpu-memory-utilization')+1],'0.93')
        single,s=self.launch('serve_c32.sh','--profile','single');self.assertEqual(single.returncode,0,single.stderr)
        self.assertEqual(s['env']['ATDEV_A4_REAL'],'had3');self.assertIn('--no-enable-prefix-caching',s['argv'])

    def test_accuracy_changes_only_w3_feature_and_sidecar_not_other_flags_or_argv(self):
        _,fast=self.launch('serve_c1.sh','--profile','single')
        run,acc=self.launch('serve_c1.sh',C1_MODE='accuracy',ATDEV_W3_SIDECAR='must-not-load')
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(acc['argv'],fast['argv'])
        expected=dict(fast['env']);expected['ATDEV_OPT']=','.join(x for x in FAST.split(',') if x!='w3_mixed');expected.pop('ATDEV_W3_SIDECAR')
        self.assertEqual(acc['env'],expected);self.assertIn('c1-accuracy',run.stderr)
        wrapper,p=self.launch('serve_c1_accuracy.sh');self.assertEqual(wrapper.returncode,0,wrapper.stderr);self.assertEqual(p,acc)
        invalid,_=self.launch('serve_c1.sh',C1_MODE='typo');self.assertNotEqual(invalid.returncode,0)

    def test_c32_only_selected_hadamard_with_builtin_mtp(self):
        run,p=self.launch('serve_c32.sh','--profile','single',ATDEV_A4_REAL='unrotated',ATDEV_DRAFTER_QUANT_PATH='must-not-load')
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

    def test_serve_sh_names_resolve_to_the_same_env_and_argv_as_the_direct_launchers(self):
        pairs=(('c1m',('serve_c1.sh','--profile','multiturn')),('c1s-accuracy',('serve_c1_accuracy.sh',)),('c1s',('serve_c1.sh','--profile','single')),
               ('c32m',('serve_c32.sh','--profile','multiturn')),('c32s',('serve_c32.sh','--profile','single')))
        for name,direct in pairs:
            a,pa=self.launch('serve.sh',name,NO_WARMUP='1',HADW='/tmp/h.safetensors');b,pb=self.launch(*direct,NO_WARMUP='1',HADW='/tmp/h.safetensors')
            self.assertEqual(a.returncode,0,a.stderr);self.assertEqual(b.returncode,0,b.stderr);self.assertEqual(pa,pb,name)
        bad,_=self.launch('serve.sh','c2m');self.assertEqual(bad.returncode,2)
        nameless,_=self.launch('serve.sh');self.assertEqual(nameless.returncode,2)
        extra,pe=self.launch('serve.sh','c1m','--port','9999',NO_WARMUP='1');self.assertEqual(extra.returncode,0,extra.stderr)
        self.assertIn('9999',pe['argv'])

    W3_ONLY=('ATDEV_W3_DQ','ATDEV_W3_DQ_MIN_M','ATDEV_W3_I8','ATDEV_W3_I8_MIN_M','ATDEV_W3_SIDECAR')

    def test_existing_launches_are_unchanged_against_the_golden_env_and_argv(self):
        golden=json.loads((ROOT/'tests'/'golden_launch_env.json').read_text().replace('{ROOT}',str(ROOT).replace('\\','/')))
        cases=(('c1m',('serve_c1.sh','--profile','multiturn')),('c1s',('serve_c1.sh','--profile','single')),
               ('c32m',('serve_c32.sh','--profile','multiturn')),('c32s',('serve_c32.sh','--profile','single')),('c1-acc-script',('serve_c1_accuracy.sh',)))
        for name,direct in cases:
            run,p=self.launch(*direct,NO_WARMUP='1',HADW='/tmp/h.safetensors');self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(p,golden[name],name)

    def test_c1m_accuracy_is_c1m_without_the_w3_overlay_and_w3_only_flags(self):
        _,mt=self.launch('serve.sh','c1m',NO_WARMUP='1')
        run,acc=self.launch('serve.sh','c1m-accuracy',NO_WARMUP='1',ATDEV_W3_SIDECAR='must-not-load');self.assertEqual(run.returncode,0,run.stderr)
        self.assertIn('c1-accuracy',run.stderr);self.assertIn('multiturn',run.stderr)
        self.assertEqual(acc['argv'],mt['argv'])
        expected={k:v for k,v in mt['env'].items() if k not in self.W3_ONLY}
        expected['ATDEV_I4_I8']='1'  # c1m-accuracy only: int8 prefill copy of the INT4 target (ATDEV_PF_GLUE=1 is shared with c1m)
        expected['ATDEV_OPT']=','.join(x for x in FAST.split(',') if x!='w3_mixed')
        self.assertEqual(acc['env'],expected)
        for k in self.W3_ONLY:self.assertNotIn(k,acc['env'],k)
        self.assertNotIn('w3_mixed',acc['env']['ATDEV_OPT'])
        for k in ('ATDEV_DFT_APC','ATDEV_PF_FUSE','ATDEV_PF_FA2','ATDEV_SWA_FINE','ATDEV_TOK_CACHE','ATDEV_GDN_SMALL','ATDEV_TREE_HSPL'):self.assertIn(k,acc['env'],k)
        self.assertEqual((acc['env']['ATDEV_I4_I8'],acc['env']['ATDEV_PF_GLUE']),('1','1'))
        for name,args in (('c1m',('serve.sh','c1m')),('c1s',('serve.sh','c1s')),('c1s-accuracy',('serve.sh','c1s-accuracy')),('c32m',('serve.sh','c32m')),('c32s',('serve.sh','c32s'))):
            r,p=self.launch(*args,NO_WARMUP='1',HADW='/tmp/h.safetensors');self.assertEqual(r.returncode,0,r.stderr);self.assertNotIn('ATDEV_I4_I8',p['env'],name)
        a=acc['argv'];self.assertIn('--enable-prefix-caching',a);self.assertEqual(a[a.index('--max-model-len')+1],'131072')
        direct,pd=self.launch('serve_c1.sh','--profile','multiturn',C1_MODE='accuracy',NO_WARMUP='1');self.assertEqual(pd['env'],acc['env']);self.assertEqual(pd['argv'],acc['argv'])
        s1,single=self.launch('serve.sh','c1s',NO_WARMUP='1');s2,sacc=self.launch('serve.sh','c1s-accuracy',NO_WARMUP='1')
        self.assertEqual(sacc['argv'],single['argv'])
        for k in self.W3_ONLY+('ATDEV_PF_GLUE',):self.assertNotIn(k,sacc['env'],k)

    DOCKER={'c1m':'c1m','c1':'c1m','c1s':'c1s','c1-single':'c1s','c1s-accuracy':'acc','c1-accuracy':'acc','c1m-accuracy':'c1m-accuracy','c32m':'c32m','c32':'c32m','c32s':'c32s','c32-single':'c32s'}

    def test_docker_entrypoint_names_and_old_aliases_dispatch_like_serve_sh(self):
        for name,target in self.DOCKER.items():
            ov=dict(ATDEV_HOME=str(ROOT),NO_WARMUP='1',HADW='/tmp/h.safetensors')
            a,pa=self.launch('docker/entrypoint.sh',name,**ov)
            b,pb=self.launch('serve_c1_accuracy.sh',**ov) if target=='acc' else self.launch('serve.sh',target,**ov)
            self.assertEqual(a.returncode,0,(name,a.stderr));self.assertEqual(b.returncode,0,b.stderr);self.assertEqual(pa,pb,name)
        extra,pe=self.launch('docker/entrypoint.sh','c1m','--port','9999',ATDEV_HOME=str(ROOT),NO_WARMUP='1');self.assertIn('9999',pe['argv'])

    def test_compose_profiles_map_to_the_intended_entrypoint_argument(self):
        try:import yaml
        except ImportError:self.skipTest('PyYAML not installed')
        svc=yaml.safe_load((ROOT/'docker-compose.yml').read_text())['services']
        seen={}
        for name,conf in svc.items():
            for prof in conf['profiles']:seen[prof]=conf['command'][0]
        expected={'c1m':'c1m','c1':'c1m','c1s':'c1s','c1-single':'c1s','c1s-accuracy':'c1s-accuracy','c1-accuracy':'c1s-accuracy','c1m-accuracy':'c1m-accuracy',
                  'c32m':'c32m','c32':'c32m','c32s':'c32s','c32-single':'c32s'}
        self.assertEqual(seen,expected)
        for arg in set(seen.values()):self.assertIn(arg,self.DOCKER)  # every command is a name the entrypoint dispatches
        dockerfile=(ROOT/'docker'/'Dockerfile').read_text()
        self.assertIn('COPY serve.sh ',dockerfile);self.assertIn('chmod +x serve.sh ',dockerfile);self.assertIn('CMD ["c1m"]',dockerfile)


if __name__=='__main__':unittest.main()
