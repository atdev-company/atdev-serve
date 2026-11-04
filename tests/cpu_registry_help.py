"""Exercise actual CLI/plugin registration; mock hardware only, never load a model."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
# The launch script sets its deployment GPU default. This helper is CPU-only.
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
import triton
from triton.backends.compiler import GPUTarget
from atdev_serve import gdn_tree_cuda

target=SimpleNamespace(get_current_target=lambda:GPUTarget('cuda',80,32),get_current_device=lambda:0,
                       get_active_torch_device=lambda:torch.device('cuda',0))
with mock.patch.object(triton.runtime.driver,'_active',target),mock.patch.object(triton.runtime.driver,'_default',target),\
     mock.patch.object(torch.cuda,'get_device_capability',return_value=(8,0)),\
     mock.patch.object(torch.cuda,'get_device_name',return_value='NVIDIA A100 CPU-only mock'),\
     mock.patch.object(torch.cuda,'get_device_properties',return_value=SimpleNamespace(name='NVIDIA A100 CPU-only mock',major=8,minor=0,multi_processor_count=108,total_memory=80*1024**3,warp_size=32)),\
     mock.patch.object(torch.cuda,'_lazy_init',side_effect=AssertionError('CPU registration must not initialize CUDA')),\
     mock.patch.object(torch.cuda,'current_device',return_value=0),\
     mock.patch.object(gdn_tree_cuda,'cuda_ext',return_value=SimpleNamespace()),\
     mock.patch('torch.utils.cpp_extension.load',return_value=SimpleNamespace()):
    from vllm.entrypoints.cli.main import main
    main()
