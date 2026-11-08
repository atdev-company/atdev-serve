"""Selected block-Hadamard W4A4 serving path (had3 only).

The deployed fused kernels/producers are ported from the measured implementation;
no fake quantization, calibration, dynamic control or unrotated serving modes.
"""
from __future__ import annotations
import functools
import json
import logging
import os
from pathlib import Path
import struct
import torch

log=logging.getLogger("atdev_serve")
_TARGET_NK={(16384,5120),(14336,5120),(5120,6144),(34816,5120),(5120,17408)}
_HEXT=[]
_HFLAG=[False]


def expected_schema():
    result={}
    for li in range(64):
        shapes={"mlp.gate_proj":(17408,5120),"mlp.up_proj":(17408,5120),"mlp.down_proj":(5120,17408)}
        if li%4==3:
            shapes.update({"self_attn.q_proj":(12288,5120),"self_attn.k_proj":(1024,5120),"self_attn.v_proj":(1024,5120),"self_attn.o_proj":(5120,6144)})
        else:
            shapes.update({"linear_attn.in_proj_qkv":(10240,5120),"linear_attn.in_proj_z":(6144,5120),"linear_attn.out_proj":(5120,6144)})
        for name,(n,k) in shapes.items():
            key=f"model.language_model.layers.{li}.{name}"
            result[key+".weight_packed"]={"dtype":"I32","shape":[n,k//8]}
            result[key+".weight_scale"]={"dtype":"BF16","shape":[n,k//128]}
    return result


def validate_header(path):
    with open(path,"rb") as f:
        length=f.read(8)
        if len(length)!=8:raise ValueError("Hadamard sidecar header missing")
        size=struct.unpack("<Q",length)[0]
        if not 2<=size<=64*1024*1024:raise ValueError("invalid Hadamard header size")
        header=json.loads(f.read(size))
    meta=header.pop("__metadata__",{})
    required={"bits":"4","group":"128","rotation":"hadamard128-blockdiag-K","src":"bf16","method":"gptq_act d0.1"}
    if any(meta.get(k)!=v for k,v in required.items()):raise ValueError("selected Hadamard INT4/group128 metadata required")
    expected=expected_schema()
    if set(header)!=set(expected):raise ValueError("Hadamard sidecar must cover exactly all400 packed/scale pairs")
    for key,spec in expected.items():
        if any(header[key].get(field)!=value for field,value in spec.items()):raise ValueError("invalid Hadamard tensor schema: "+key)
    return set(expected)


def resolve_sidecar(spec):
    from atdev_serve.hub import resolve_files
    paths=resolve_files(spec)
    if len(paths)!=1:raise ValueError("exactly one Hadamard sidecar file required")
    path=str(Path(paths[0]).expanduser().resolve())
    if not os.path.isfile(path):raise FileNotFoundError(path)
    validate_header(path)
    return path


def _wrap_loader(path):
    from safetensors import safe_open
    from vllm.model_executor.model_loader import default_loader as dl
    orig=dl.DefaultModelLoader._get_weights_iterator
    if getattr(orig,"_atdev_hadw",False):
        if orig._atdev_hadw_path!=path:raise ValueError("Hadamard sidecar changed after registration")
        return
    keys=validate_header(path);reader=safe_open(path,"pt",device="cpu")
    def iterator(self,source):
        seen=set()
        for name,tensor in orig(self,source):
            if name in keys:
                if name in seen:raise ValueError("duplicate Hadamard checkpoint tensor: "+name)
                value=reader.get_tensor(name)
                if value.shape!=tensor.shape or value.dtype!=tensor.dtype:raise ValueError("Hadamard/checkpoint shape or dtype mismatch: "+name)
                if name.endswith(".weight_scale") and (not torch.isfinite(value).all() or (value<=0).any()):raise ValueError("invalid Hadamard BF16 scales")
                seen.add(name);yield name,value
            else:yield name,tensor
        if seen!=keys:raise ValueError("checkpoint did not load every Hadamard tensor")
        log.warning("atdev Hadamard: loaded %d projections/%d tensors, bits4 group128, had3",len(seen)//2,len(seen))
    iterator._atdev_hadw=True;iterator._atdev_hadw_path=path
    dl.DefaultModelLoader._get_weights_iterator=iterator


class GDNAdapter:
    """Preserve release BF16-state ABI while calling the exact selected kernel."""
    def __init__(self,module):self._module=module
    def __getattr__(self,name):return getattr(self._module,name)
    # Two caller generations: the single-turn release callers pass the short ABI (23 / 27 arguments, bf16 state implied),
    # the multi-turn gdn_single_state / gdn_a4_fold callers already pass the full ABI of the kernel (26 / 30 arguments).
    def gdn_ss(self,*args):
        if len(args)==26:return self._module.gdn_ss(*args)
        if len(args)!=23:raise ValueError("unexpected release GDN ABI")
        return self._module.gdn_ss(*args[:17],False,1,*args[17:],0)
    def gdn_ss_a4(self,*args):
        if len(args)==30:return self._module.gdn_ss_a4(*args)
        if len(args)!=27:raise ValueError("unexpected release folded GDN ABI")
        return self._module.gdn_ss_a4(*args[:17],False,1,*args[17:23],0,*args[23:])


def had_ext():
    if not _HEXT:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "acc_c32", "a4_had.cu")
        _HEXT.append(load(name="atdev_acc_a4had", sources=[src], verbose=False,
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"]))
    return _HEXT[0]

def _register_real_had2(strict: bool = True) -> None:
    """ATDEV_A4_REAL=had2: as had, but the a4 path quantises with csrc/acc_c32/a4_had.cu quant_a4_had (rotation
    fused into the s4 quant: no extra launch) and the other paths rotate with its had_rot kernel. The rotation
    flag is set only around the target linears' mp_linear calls, so other quant_a4 users (A4SCR screen) are
    unchanged."""
    from atdev_serve import marlin_plus as mp
    orig = mp.mp_linear
    if getattr(orig, "_atdev_had", False):
        return
    orig_mod_a4 = mp.mod_a4
    flag = _HFLAG

    class _Proxy:
        def __init__(self, m):
            self._m = m

        def __getattr__(self, k):
            return getattr(self._m, k)

        def quant_a4(self, x, gs, a_pk, a_s, tmb):
            if flag[0]:
                flag[0] = False
                return had_ext().quant_a4_had(x, gs, a_pk, a_s, tmb)
            return self._m.quant_a4(x, gs, a_pk, a_s, tmb)

    px = []

    def mod_a4():
        if not px:
            px.append(_Proxy(orig_mod_a4()))
        return px[0]

    @functools.wraps(orig)  # custom-op schema from the original annotations
    def mp_linear(x, mq, ms, gs, n, k, a8):
        m = x.shape[0]
        if a8 and (n, k) in _TARGET_NK and m > 0:
            ent = mp.lookup(n, k, m, a8)
            if (ent is not None and ent.get("a4") and m <= 16 * ent["cfg"][4] and k <= mp.A4_MAX_K
                    and ent["cfg"][4] <= mp.A4_MAX_TMB):
                flag[0] = True
                try:
                    return orig(x, mq, ms, gs, n, k, a8)
                finally:
                    consumed = not flag[0]
                    flag[0] = False  # (had3: the gdn_a4_fold proxy may skip quant_a4 for folded inputs)
                    assert consumed or not strict, "a4 path did not consume the rotation flag"
            x = had_ext().had_rot(x if x.stride(-1) == 1 and x.stride(0) % 4 == 0 else x.contiguous())
        return orig(x, mq, ms, gs, n, k, a8)

    mp_linear._atdev_had = True
    mp.mp_linear = mp_linear
    mp.mod_a4 = mod_a4
    log.warning("atdev a4_sim: REAL had2 path (fused rotate+s4 quant, had_rot elsewhere) for %s", sorted(_TARGET_NK))

def _had3_pack_store():
    """fused_quant._a4_pack_store with the group rotation (bf16 x +-1 H128 tensor-core dot, fp32 accumulate:
    exact products of the bf16-valued producer outputs) before the unchanged s4 quant / layout. NGB >= 16."""
    import triton
    import triton.language as tl

    @triton.jit
    def _a4_pack_store(y, row, g0, NGB: tl.constexpr, ngroups, apk_ptr, as_ptr, gs, m_pad, tmb,
                       MK: tl.constexpr = True):
        y0 = tl.reshape(y, [NGB, 128])
        hi = tl.arange(0, 128)[:, None] & tl.arange(0, 128)[None, :]
        hi = hi ^ (hi >> 4)
        hi = hi ^ (hi >> 2)
        hi = hi ^ (hi >> 1)
        hm = (1 - 2 * (hi & 1)).to(tl.bfloat16)
        yy = tl.dot(y0.to(tl.bfloat16), hm, out_dtype=tl.float32) * 0.08838834764831845
        amax = tl.max(tl.abs(yy), axis=1)
        sc = tl.maximum(amax, 1e-8) * (1.0 / 7.0)
        if MK:
            rc = tl.extra.cuda.libdevice.div_rn(1.0, sc)
            scb = tl.broadcast_to(sc[:, None], yy.shape)
            rcb = tl.broadcast_to(rc[:, None], yy.shape)
            q1 = tl.inline_asm_elementwise(
                "{ .reg .f32 q0, rr, ns; mul.rn.ftz.f32 q0, $1, $3; neg.f32 ns, $2; "
                "fma.rn.ftz.f32 rr, ns, q0, $1; fma.rn.ftz.f32 $0, rr, $3, q0; }",
                "=r,r,r,r", [yy, scb, rcb], dtype=tl.float32, is_pure=True, pack=1)
            q = tl.extra.cuda.libdevice.rint(q1)
        else:
            q = tl.extra.cuda.libdevice.rint(tl.extra.cuda.libdevice.div_rn(yy, sc[:, None]))
        q = tl.minimum(tl.maximum(q, -8.0), 7.0).to(tl.int32) & 15
        q5 = tl.reshape(q, [NGB, 4, 2, 4, 4])
        p_i = tl.arange(0, 2)[None, None, :, None, None]
        e_i = tl.arange(0, 4)[None, None, None, None, :]
        w = tl.sum(tl.sum(q5 << (8 * e_i + 4 * p_i), axis=4), axis=2)
        g = g0 + tl.arange(0, NGB)[:, None, None]
        a_i = tl.arange(0, 4)[None, :, None]
        b_i = tl.arange(0, 4)[None, None, :]
        kb = 2 * g + a_i // 2
        l_i = (row % 8) * 4 + b_i
        r_i = (row % 16) // 8 + 2 * (a_i % 2)
        idx = ((kb * tmb + row // 16) * 32 + l_i) * 4 + r_i
        tl.store(apk_ptr + idx, w, mask=g < ngroups)
        gv = g0 + tl.arange(0, NGB)
        tl.store(as_ptr + gv * m_pad + row, sc * gs, mask=gv < ngroups)

    return _a4_pack_store

_FUSE = os.environ.get("ATDEV_A4ACC_FUSE", "0") == "1"
_REXT = []


def rot_ext():
    if not _REXT:
        from torch.utils.cpp_extension import load
        src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "a4acc", "rot_i8.cu")
        _REXT.append(load(name="atdev_a4acc_rot_i8s4", sources=[src], verbose=False,
                          extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"]))
    return _REXT[0]


def _rq(fn, args, gs, n_next, k, m):
    """A4ACC: fused rotate+int8 producer; when the consumer will run PF_A4 it also emits the s4 fragments straight from
    the rotated bf16 values (ATDEV_A4ACC_S4=1) and flags them in marlin_plus.PF_A4_READY."""
    from atdev_serve import marlin_plus as mp
    mp.PF_A4_READY[0] = None
    g = gs.reshape(1).float()
    if mp.pf_a4_emit_ok(n_next, k, m):
        b = mp.bufs(args[0].device)
        q, sc = fn(*args, g, b["a4_pk_big"], b["a4_s_big"], mp.a4_clip())
        mp.PF_A4_READY[0], mp.PF_A4_READY[1] = q.data_ptr(), m
        return q, sc
    return fn(*args, g)


def _int8_rot(y_bf16: torch.Tensor, gs: torch.Tensor, n_next: int = 0):
    """rotated per-token int8 (the int8 producers' formula on x H): q, s * input_global_scale"""
    if _FUSE:
        y = y_bf16 if (y_bf16.stride(-1) == 1 and y_bf16.stride(0) % 4 == 0) else y_bf16.contiguous()
        return _rq(rot_ext().rot_quant, (y,), gs, n_next, y.shape[1], y.shape[0])
    from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8
    yr = had_ext().had_rot(y_bf16.contiguous())
    q, s = per_token_quant_int8(yr)
    return q, (s.view(-1) * gs.reshape(1).float()).contiguous()

def _register_real_had3() -> None:
    """ATDEV_A4_REAL=had3: the FULL c=32 stack (fused_quant / gdn_a4_fold / c32_prod / c32c csum producers) on
    rotated weights. Every s4 producer quantises x H: Triton _a4_pack_store -> _had3_pack_store, c32_prod and
    c32c_csum -> their generated *_h.cu copies (scripts/acc_c32_mkhad.py), the GDN fold -> ATDEV_GDN_SRC
    gdn_ss12h/15h.cu (checked here); int8 producers (non-a4 M) -> bf16 producer output + had_rot + per-token
    int8; remaining bf16 / quant_a4 paths -> the had2 mp_linear wrapper."""
    import triton  # noqa: F401
    from atdev_serve import c32_prod, c32c_csum, fused_quant as fq, marlin_plus as mp
    from atdev_serve.register import enabled
    assert not getattr(mp, "U8", False), "had3: ATDEV_MS_U8 (int8 code sums) not supported"
    src = os.environ.get("ATDEV_GDN_SRC", "")
    if "gdn_a4_fold" in enabled():
        assert src.endswith("h.cu"), f"had3 + gdn_a4_fold needs a generated gdn_ss*h.cu (ATDEV_GDN_SRC={src})"
    os.environ.setdefault("ATDEV_FQ_A4_NORM", "16,8")
    os.environ.setdefault("ATDEV_FQ_A4_SILU", "16,4")
    for k in ("ATDEV_FQ_A4_NORM", "ATDEV_FQ_A4_SILU"):
        assert int(os.environ[k].split(",")[0]) >= 16, f"had3: {k} NGB must be >= 16 (tl.dot)"
    for k in ("ATDEV_C32_NE", "ATDEV_C32H_SILU"):
        assert os.environ.get(k, "0") == "0", f"had3: {k} producers are not rotated"
    _register_real_had2(strict=False)
    fq._a4_pack_store = _had3_pack_store()
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
    from torch.utils.cpp_extension import load
    flags = ["-O3", "-gencode=arch=compute_80,code=sm_80", "-std=c++17"]
    pe, ce = {}, {}

    def prod_ext(mk=None):
        mk = True if mk is None else mk
        if mk not in pe:
            pe[mk] = load(name="atdev_c32_prod_h" + ("_mk" if mk else ""), sources=[os.path.join(cdir, "c32_prod_h.cu")],
                          verbose=False, extra_cuda_cflags=flags + (["-DMK=1"] if mk else []))
        return pe[mk]

    def csum_ext(mk=None):
        if mk is None:
            mk = os.environ.get("ATDEV_C32_MK", "0") == "1"
        if mk not in ce:
            ce[mk] = load(name="atdev_c32c_csum_h" + ("_mk" if mk else ""), sources=[os.path.join(cdir, "c32c_csum_h.cu")],
                          verbose=False, extra_cuda_cflags=flags + (["-DMK=1"] if mk else []))
        return ce[mk]

    c32_prod.ext = prod_ext
    c32c_csum.ext = csum_ext
    o_norm, o_silu, o_gate, o_cs = fq.norm_quant, fq.silu_quant, fq.gate_quant, fq.cs_mp_linear

    @functools.wraps(o_norm)
    def norm_quant(x, residual, weight, eps, gs, want_bf16, n_next):
        m, n = x.shape
        if m == 0 or fq._a4_ent(n_next, n, m) is not None:
            return o_norm(x, residual, weight, eps, gs, want_bf16, n_next)
        xb, _, _, ro = o_norm(x, residual, weight, eps, gs, True, n_next)
        xq, xs = _int8_rot(xb, gs, n_next)
        return (xb if want_bf16 else xb[:0]), xq, xs, ro

    @functools.wraps(o_silu)
    def silu_quant(gu, gs, *rest):  # (gu, gs, n_next) or, in fused_quant, (gu, gs, cfg, n_next)
        n_next = rest[-1] if rest else 0
        m, n2 = gu.shape
        n = n2 // 2
        if m == 0 or (n_next and fq._a4_ent(n_next, n, m) is not None):
            return o_silu(gu, gs, *rest)
        if _FUSE:
            return _rq(rot_ext().silu_rot_quant, (gu,), gs, n_next, n, m)
        g, u = gu[:, :n].float(), gu[:, n:].float()
        return _int8_rot((g * torch.sigmoid(g) * u).to(torch.bfloat16), gs)

    @functools.wraps(o_gate)
    def gate_quant(a, g, gs, n_next=0):
        m, n = a.shape
        if m == 0 or (n_next and fq._a4_ent(n_next, n, m) is not None):
            return o_gate(a, g, gs, n_next)
        if _FUSE:
            return _rq(rot_ext().gate_rot_quant, (a, g), gs, n_next, n, m)
        return _int8_rot((a.float() * torch.sigmoid(g.float())).to(torch.bfloat16), gs)

    @functools.wraps(o_cs)
    def cs_mp_linear(x, w_q, w_s, gs, n, k):
        # A4ACC: the flag asks the proxy quant_a4 to rotate; set it only when cs_mp_linear takes its own small-M
        # quant_a4 path. For large M it falls back to mp_linear (which rotates via had_rot) and a set flag would
        # make PF_A4's quant_a4 rotate a second time.
        _ent = mp.lookup(n, k, x.shape[0], True) if x.shape[0] > 0 else None
        _HFLAG[0] = (n, k) in _TARGET_NK and _ent is not None and bool(_ent.get("a4")) and x.shape[0] <= 16 * _ent["cfg"][4]
        try:
            return o_cs(x, w_q, w_s, gs, n, k)
        finally:
            _HFLAG[0] = False

    fq.norm_quant, fq.silu_quant, fq.gate_quant, fq.cs_mp_linear = norm_quant, silu_quant, gate_quant, cs_mp_linear
    log.warning("atdev a4_sim: REAL had3 path (rotated s4/int8 producers, GDN %s)", src)

def register():
    if os.environ.get("ATDEV_A4_REAL")!="had3":raise ValueError("release a4_sim supports only selected had3")
    if os.environ.get("ATDEV_C32_MK") != "1":raise ValueError("selected had3 requires ATDEV_C32_MK=1")
    spec=os.environ.get("ATDEV_A4_HADW","")
    if not spec:raise ValueError("selected Hadamard sidecar required")
    path=resolve_sidecar(spec)
    os.environ["ATDEV_A4_HADW"]=path
    _wrap_loader(path)
    _register_real_had3()
