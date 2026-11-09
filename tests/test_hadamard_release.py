"""CPU loader schema/resolution and compatibility with release helper APIs."""
import importlib
import json
import os
from pathlib import Path
import struct
import re
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from atdev_serve import a4_sim as a4

class Hadamard(unittest.TestCase):
    def test_hub_resolves_one_file_then_full_schema_rejects_incomplete_or_wrong(self):
        self.assertTrue(callable(getattr(a4,'resolve_sidecar',None)))
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'had.safetensors'
            header=a4.expected_schema();header['__metadata__']={'bits':'4','group':'128','rotation':'hadamard128-blockdiag-K','src':'bf16','method':'gptq_act d0.1'}
            def write(h):
                raw=json.dumps(h).encode();path.write_bytes(struct.pack('<Q',len(raw))+raw)
            write(header)
            with mock.patch('atdev_serve.hub.resolve_files',return_value=(str(path),)) as resolve:
                self.assertEqual(a4.resolve_sidecar('org/repo:hadamard/file.safetensors'),str(path.resolve()))
                resolve.assert_called_once_with('org/repo:hadamard/file.safetensors')
            self.assertEqual(len(a4.validate_header(path)),800)
            with mock.patch('atdev_serve.hub.resolve_files',return_value=(str(path),str(path))):
                with self.assertRaises(ValueError):a4.resolve_sidecar('org/repo:folder/')
            for fault in ('missing','shape','dtype','metadata'):
                h=json.loads(json.dumps(header));key='model.language_model.layers.0.mlp.down_proj.weight_packed'
                if fault=='missing':h.pop(key)
                elif fault=='shape':h[key]['shape']=[1,1]
                elif fault=='dtype':h[key]['dtype']='F32'
                else:h['__metadata__']['rotation']='none'
                write(h)
                with self.assertRaises(ValueError,msg=fault):a4.validate_header(path)

    def test_gdn_selected_abi_inserts_bf16_state_defaults_without_changing_old_args(self):
        self.assertTrue(callable(getattr(a4,'GDNAdapter',None)))
        raw=SimpleNamespace(gdn_ss=mock.Mock(),gdn_ss_a4=mock.Mock());adapter=a4.GDNAdapter(raw)
        args=tuple(range(23));adapter.gdn_ss(*args)
        self.assertEqual(raw.gdn_ss.call_args.args,args[:17]+(False,1)+args[17:]+(0,))
        extended=tuple(range(27));adapter.gdn_ss_a4(*extended)
        self.assertEqual(raw.gdn_ss_a4.call_args.args,extended[:17]+(False,1)+extended[17:23]+(0,)+extended[23:])
        source=(Path(a4.__file__).parent/'csrc/gdn_var/gdn_ss12h.cu').read_text()
        for name,count in (('gdn_ss',26),('gdn_ss_a4',30)):
            declaration=re.search(r'\b(?:bool|void) '+name+r'\((.*?)\)\s*\{',source,re.S)
            self.assertEqual(len(declaration.group(1).split(',')),count)

    def test_gdn_adapter_passes_the_full_abi_of_the_multiturn_callers_through_unchanged(self):
        raw=SimpleNamespace(gdn_ss=mock.Mock(),gdn_ss_a4=mock.Mock());adapter=a4.GDNAdapter(raw)
        full=tuple(range(26));adapter.gdn_ss(*full);self.assertEqual(raw.gdn_ss.call_args.args,full)
        full=tuple(range(30));adapter.gdn_ss_a4(*full);self.assertEqual(raw.gdn_ss_a4.call_args.args,full)
        for bad in (22,24,25,27):
            with self.assertRaises(ValueError):adapter.gdn_ss(*range(bad))
        for bad in (26,28,29,31):
            with self.assertRaises(ValueError):adapter.gdn_ss_a4(*range(bad))
        gv=Path(a4.__file__).parent/'csrc/gdn_var'
        for name in ('gdn_ss12h.cu','gdn_ss15h.cu'):  # both selected kernels declare the full ABI the callers use
            source=(gv/name).read_text()
            for fn,count in (('gdn_ss',26),('gdn_ss_a4',30)):
                declaration=re.search(r'\b(?:bool|void) '+fn+r'\((.*?)\)\s*\{',source,re.S)
                self.assertEqual(len(declaration.group(1).split(',')),count,(name,fn))

    def test_gdn_extension_hook_accepts_both_selected_kernels_and_wraps_them(self):
        from atdev_serve import gdn_single_state as gss
        gv=Path(gss.__file__).parent/'csrc'/'gdn_var'
        for name,ok in (('gdn_ss15h.cu',True),('gdn_ss12h.cu',True),('../gdn_ss.cu',False)):
            env={'ATDEV_A4_REAL':'had3','ATDEV_OPT':'a4_sim,gdn_single_state','ATDEV_GDN_SRC':str(gv/name)}
            with mock.patch.dict(os.environ,env),mock.patch.object(gss,'_EXT',None),\
                 mock.patch('torch.utils.cpp_extension.load',return_value=SimpleNamespace(gdn_ss=mock.Mock())) as load:
                if ok:
                    ext=gss.cuda_ext();self.assertIsInstance(ext,a4.GDNAdapter)
                    self.assertTrue(load.call_args.kwargs['sources'][0].endswith(name))
                else:
                    with self.assertRaises(ValueError):gss.cuda_ext()

    def test_loader_substitution_refuses_missing_duplicate_invalid_or_mismatched(self):
        import torch
        from types import ModuleType
        modules={name:ModuleType(name) for name in ('vllm','vllm.model_executor','vllm.model_executor.model_loader')}
        key='model.language_model.layers.0.mlp.down_proj'
        keys={key+'.weight_packed',key+'.weight_scale'}
        values={key+'.weight_packed':torch.zeros(2,32,dtype=torch.int32),key+'.weight_scale':torch.ones(2,2).bfloat16()}
        class Loader:
            def _get_weights_iterator(self,source):yield from source
        dl=SimpleNamespace(DefaultModelLoader=Loader)
        modules['vllm.model_executor.model_loader'].default_loader=dl
        reader=SimpleNamespace(get_tensor=lambda name:values[name])
        with mock.patch.dict(sys.modules,modules),mock.patch.object(a4,'validate_header',return_value=keys),mock.patch('safetensors.safe_open',return_value=reader):
            a4._wrap_loader('/single.safetensors')
            source=[(name,value.clone()) for name,value in values.items()]
            result=list(Loader()._get_weights_iterator(source));self.assertEqual({name for name,_ in result},keys)
            for bad in (source[:1],source+[source[0]],[(source[0][0],torch.zeros(1,dtype=torch.int32)),source[1]]):
                with self.assertRaises(ValueError):list(Loader()._get_weights_iterator(bad))
            values[key+'.weight_scale'].fill_(float('nan'))
            with self.assertRaises(ValueError):list(Loader()._get_weights_iterator(source))

    def test_selected_producer_wrapper_matches_release_three_argument_silu_api(self):
        import torch
        import atdev_serve
        from atdev_serve import register
        original=mock.Mock(return_value=('q','s'))
        def silu(gu,gs,n_next=0):return original(gu,gs,n_next)
        fq=SimpleNamespace(norm_quant=mock.Mock(),silu_quant=silu,gate_quant=mock.Mock(),cs_mp_linear=mock.Mock(),_a4_ent=lambda *a:None)
        mp=SimpleNamespace()
        with mock.patch.object(atdev_serve,'fused_quant',fq,create=True),mock.patch.object(atdev_serve,'marlin_plus',mp,create=True),mock.patch.object(atdev_serve,'c32_prod',SimpleNamespace(),create=True),mock.patch.object(atdev_serve,'c32c_csum',SimpleNamespace(),create=True),mock.patch.object(a4,'_register_real_had2'),mock.patch.object(a4,'_had3_pack_store',return_value='kernel'),mock.patch.object(register,'enabled',return_value={'gdn_a4_fold'}),mock.patch.dict(os.environ,{'ATDEV_GDN_SRC':'gdn_ss12h.cu','ATDEV_C32_MK':'1','ATDEV_FQ_A4_NORM':'16,8','ATDEV_FQ_A4_SILU':'16,4'}):
            a4._register_real_had3()
            empty=torch.empty(0,256);gs=torch.ones(1)
            self.assertEqual(fq.silu_quant(empty,gs,17408),('q','s'))
            self.assertEqual(original.call_args.args[2],17408)

if __name__=='__main__':unittest.main()
