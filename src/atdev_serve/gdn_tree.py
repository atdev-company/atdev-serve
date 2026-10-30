"""Tree verification kernels for the hybrid GDN target (ATDEV_OPT=dflash_tree).

Composes with A3 ``gdn_single_state``: same slots, same token-log layout, same
parity double buffer, and -- after the per-step *compaction* below -- exactly
A3's replay semantics, so every stock / A3 code path (prefill, mixed batches,
the A3 dispatcher) stays valid between tree steps.

Per request and step the verify batch holds the N nodes of a token tree in
topological order (node 0 = root = last accepted token). Node i's state is its
parent's state updated with token i:

    S_i = a_i S_par(i) (I - b_i k_i k_i^T) + b_i v_i k_i^T,   o_i = S_i q_i

(``a`` = exp(g) decay, ``b`` = beta). Unrolled along the ancestor chain of i
from the pre-root state S_0 (the ancestor-masked chunk / "TreeWY" form):

    G_i = sum_{j anc-or-self of i} g_j
    W   = (I + A)^-1 R,   A_ij = b_i e^{G_i-G_j} (k_i.k_j)  (j strict anc of i)
                          R_i  = b_i (v_i - e^{G_i} S_0 k_i)
    o_i = e^{G_i} S_0 q_i + sum_{j anc-or-self i} e^{G_i-G_j} (q_i.k_j) w_j

(I + A) is unit lower triangular in topological order, A is nilpotent, so
(I + A)^-1 = (I - A)(I + A^2)(I + A^4)(I + A^8) for N <= 16: six 16x16x16
products instead of an N-step dependent row solve (the old kernel's solve loop
was the bottleneck: 67-195 us/layer at N=32). All products run in IEEE fp32
(``input_precision="ieee"``, FMA, no TF32 anywhere by default).

Step protocol (per GDN layer, request with state slots s0 = indices[n,0],
s1 = indices[n,1], a = num_accepted >= 1 from the previous step):
  1. h <- slot s0 (state after the previous root); replay log entries
     0..a-2 sequentially (identical instructions to A3's kernel),
  2. tree pass above; outputs for all N nodes,
  3. slot s0 <- state after the root (the single state write),
  4. log buffer (1-p) <- (k, v, decay, beta) of nodes 1..N-1 (entry = node-1),
     parity flips.
After the target's acceptance (path = root..a-1 node indices) the runner
launches ``compact_gdn_log`` once for all 48 layers: entries path[d]-1 move to
d-1, so the log is in A3's sequential order and step 1 above (and A3 / the
dispatcher in mixed batches) replays the right tokens. For a chain path the
compaction is a no-op.

Conv1d (width 4): stock keeps [h2, h3, x_0..x_{T-1}] per conv slot and reads
the window at offset a-1. ``conv_tree_update`` reads the same window and
convolves node i with (3rd, 2nd, 1st ancestor, i); ``compact_conv_state``
moves x_{path[d]} to column 2+d after acceptance, restoring stock layout.

Full-attention layers: vLLM's TRITON_ATTN ``unified_attention`` takes a
``qq_bias`` [N, N] (0 / -inf ancestor mask); K/V of node i is written at the
slot of position pos0+i and ``compact_kv`` moves path nodes to pos0+d.

Also here: the GPU tree builder (best-first k-best paths over DFlash2's CRF
lattice, same algorithm as ``tree_spec.build_tree``), greedy tree acceptance,
and the tiny per-step activation / position kernels used by
``atdev_serve.dflash_tree``.

Files: this module, atdev_serve/dflash_tree.py (vLLM glue),
tests/test_gdn_tree.py, docs/research/tree-verify.md.
"""

from __future__ import annotations

import os

import torch

from atdev_serve.gdn_single_state import L2NORM_EPS, log_layout, slot_capacity_f32

try:
    import triton
    import triton.language as tl

    from atdev_serve.gdn_single_state import _gdn_rank1_step
except ImportError:  # pragma: no cover
    triton = None
    tl = None


# ---------------------------------------------------------------------------
# Torch references
# ---------------------------------------------------------------------------


def ancestor_matrix(parent: torch.Tensor) -> torch.Tensor:
    """parent [N] (root = -1) -> M [N, N] bool, M[i, j] = j anc-or-self of i."""
    n = parent.numel()
    m = torch.zeros(n, n, dtype=torch.bool)
    par = parent.tolist()
    for i in range(n):
        j = i
        while j >= 0:
            m[i, j] = True
            j = par[j]
    return m


def anc_bits_from_parent(parent: torch.Tensor) -> torch.Tensor:
    """int64 bitmask per node (N <= 63) of its ancestors-or-self."""
    m = ancestor_matrix(parent)
    w = (1 << torch.arange(m.shape[1], dtype=torch.int64))
    return (m.to(torch.int64) * w).sum(1)


def ref_rank1(S, k, v, decay, beta):
    """S [..., V, K] -> state after one token (same op order as the kernels)."""
    S = S * decay[..., None, None]
    delta = (v - (S * k[..., None, :]).sum(-1)) * beta[..., None]
    return S + delta[..., :, None] * k[..., None, :]


def ref_tree_scan(q, k, v, decay, beta, parent, S0):
    """Ground truth: per-node recurrence. Shapes (one head or batched heads
    in leading dims): q, k [N, ..., K]; v [N, ..., V]; decay, beta [N, ...];
    S0 [..., V, K]. Returns o [N, ..., V], states [N, ..., V, K]."""
    n = q.shape[0]
    states = [None] * n
    o = torch.empty(v.shape, dtype=S0.dtype)
    for i in range(n):
        p = int(parent[i])
        Sp = S0 if p < 0 else states[p]
        Si = ref_rank1(Sp, k[i], v[i], decay[i], beta[i])
        states[i] = Si
        o[i] = (Si * q[i][..., None, :]).sum(-1)
    return o, torch.stack(states)


def ref_tree_wy(q, k, v, decay, beta, parent, S0):
    """The ancestor-masked closed form, one head: q, k [N, K]; v [N, V];
    S0 [V, K]. Returns o [N, V], w [N, V], G [N]."""
    M = ancestor_matrix(parent).to(S0.dtype)
    Ms = M - torch.eye(M.shape[0], dtype=S0.dtype)
    G = M @ torch.log(decay)
    ratio = torch.exp((G[:, None] - G[None, :]) * M) * M  # 0 off-ancestor
    A = Ms * ratio * (k @ k.T)
    rhs = beta[:, None] * (v - torch.exp(G)[:, None] * (k @ S0.T))
    w = torch.linalg.solve_triangular(torch.eye(len(G), dtype=S0.dtype) + beta[:, None] * A,
                                      rhs, upper=False)
    o = torch.exp(G)[:, None] * (q @ S0.T) + (M * ratio * (q @ k.T)) @ w
    return o, w, G


def ref_state_from_wy(path, k, w, G, S0):
    """S after the last node of ``path`` (root..a) from the WY factors."""
    a = path[-1]
    S = torch.exp(G[a]) * S0
    for j in path:
        S = S + torch.exp(G[a] - G[j]) * torch.outer(w[j], k[j])
    return S


def ref_token_params(q, k, a, b, A_log, dt_bias, scale, sp_beta=1.0, sp_thr=20.0):
    dt = q.dtype
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + L2NORM_EPS) * scale
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + L2NORM_EPS)
    x = a.to(dt) + dt_bias.to(dt)
    sp = torch.where(sp_beta * x <= sp_thr, torch.log1p(torch.exp(sp_beta * x)) / sp_beta, x)
    decay = torch.exp(-torch.exp(A_log.to(dt)) * sp)
    beta = torch.sigmoid(b.to(dt))
    return q, k, decay, beta


def ref_conv_tree(x, weight, bias, window, conv_src, activation=True):
    """x [N, C]; weight [C, 4]; window [3, C] (oldest first); conv_src [N, 3]
    (1st, 2nd, 3rd ancestor; -1..-3 = window[2], window[1], window[0])."""
    n = x.shape[0]
    out = torch.empty_like(x)
    for i in range(n):
        cols = []
        for s in range(3):
            j = int(conv_src[i, s])
            cols.append(x[j] if j >= 0 else window[3 + j])
        y = (weight[:, 3] * x[i] + weight[:, 2] * cols[0] + weight[:, 1] * cols[1]
             + weight[:, 0] * cols[2])
        if bias is not None:
            y = y + bias
        out[i] = torch.nn.functional.silu(y) if activation else y
    return out


def conv_window_index(prev_path, num_accepted):
    """Indices into the stored conv array [h2, h3, x_0..x_{T-1}] of the 3
    window columns (oldest first) after accepting prev_path[:a]."""
    a = int(num_accepted)
    idx = []
    for j in (a - 3, a - 2, a - 1):
        idx.append(2 + int(prev_path[j]) if j >= 0 else 2 + j)
    return idx




# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------

if triton is not None:

    @triton.jit
    def _gdn_tree_kernel(
        q_ptr, k_ptr, v_ptr, stride_q, stride_k, stride_v,
        a_ptr, b_ptr, stride_a, stride_b,
        A_log_ptr, dt_bias_ptr,
        state_ptr, log_ptr, par_ptr,
        slot_state, slot_log, slot_par,
        si_ptr, stride_si, cu_ptr, acc_ptr,
        anc_ptr, stride_anc,
        o_ptr, stride_o,
        z_ptr, stride_z, nw_ptr, tick_ptr, eps,
        scale, sp_beta, sp_threshold,
        H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
        BV: tl.constexpr, NP: tl.constexpr, LOG2NP: tl.constexpr,
        MAX_LOG: tl.constexpr, LOG_BASE: tl.constexpr, BUF: tl.constexpr,
        V_OFF: tl.constexpr, D_OFF: tl.constexpr, B_OFF: tl.constexpr,
        IP: tl.constexpr, IP_S: tl.constexpr, BF16X3: tl.constexpr,
        FUSE_NORM: tl.constexpr, SIGMOID: tl.constexpr,
    ):
        n = tl.program_id(0)
        hv = tl.program_id(1)
        iv = tl.program_id(2)
        NV: tl.constexpr = V // BV
        G: tl.constexpr = HV // H
        hk = hv // G

        bos = tl.load(cu_ptr + n).to(tl.int64)
        eos = tl.load(cu_ptr + n + 1).to(tl.int64)
        T = eos - bos
        if T <= 0:
            return
        ri = tl.arange(0, NP)  # node index along axis 0
        cj = tl.arange(0, NP)  # node index along axis 1
        offs_k = tl.arange(0, K)
        offs_v = iv * BV + tl.arange(0, BV)
        rmask = ri < T
        cmask = cj < T
        tok_r = bos + ri
        tok_c = bos + cj
        o_ptrs = o_ptr + tok_c[None, :] * stride_o + hv * V + offs_v[:, None]  # [BV, NP]
        s0 = tl.load(si_ptr + n * stride_si).to(tl.int64)
        s1 = tl.load(si_ptr + n * stride_si + 1).to(tl.int64)
        if (s0 <= 0) | (s1 <= 0) | (T > MAX_LOG + 1):
            # padded / NULL rows: zero output, like the stock kernels
            tl.store(o_ptrs, tl.zeros([BV, NP], dtype=tl.float32).to(o_ptr.dtype.element_ty),
                     mask=cmask[None, :])
            return

        # ---- 1) pre-root state: slot 0 + sequential replay (== A3 kernel)
        n_acc = tl.load(acc_ptr + n)
        n_rep = tl.minimum(tl.maximum(n_acc - 1, 0), MAX_LOG)
        par_addr = par_ptr + s1 * slot_par + hv * NV + iv
        p = tl.load(par_addr) & 1
        p = tl.where(n_rep > 0, p, 0)
        lbase = log_ptr + s1 * slot_log + LOG_BASE
        rbuf = lbase + p * BUF
        wbuf = lbase + (1 - p) * BUF
        h_ptrs = (state_ptr + s0 * slot_state + hv * V * K
                  + offs_v[:, None] * K + offs_k[None, :])
        h = tl.load(h_ptrs).to(tl.float32)
        for t in range(0, n_rep):
            kk = tl.load(rbuf + (t * H + hk) * K + offs_k)
            vv = tl.load(rbuf + V_OFF + (t * HV + hv) * V + offs_v)
            dd = tl.load(rbuf + D_OFF + t * HV + hv)
            bb = tl.load(rbuf + B_OFF + t * HV + hv)
            h = _gdn_rank1_step(h, kk, vv, dd, bb)

        # ---- 2) per-node parameters
        q_raw = tl.load(q_ptr + tok_r[:, None] * stride_q + hk * K + offs_k[None, :],
                        mask=rmask[:, None], other=0.0)
        k_raw = tl.load(k_ptr + tok_r[:, None] * stride_k + hk * K + offs_k[None, :],
                        mask=rmask[:, None], other=0.0)
        qf = q_raw.to(tl.float32)
        kf = k_raw.to(tl.float32)
        rq = tl.rsqrt(tl.sum(qf * qf, axis=1) + 1e-6)
        rk = tl.rsqrt(tl.sum(kf * kf, axis=1) + 1e-6)
        q = qf * rq[:, None]
        q = q * scale
        k = kf * rk[:, None]
        vt = tl.load(v_ptr + tok_c[None, :] * stride_v + hv * V + offs_v[:, None],
                     mask=cmask[None, :], other=0.0).to(tl.float32)  # [BV, NP]
        A = tl.exp(tl.load(A_log_ptr + hv).to(tl.float32))
        dt_b = tl.load(dt_bias_ptr + hv).to(tl.float32)
        # same arithmetic in both layouts -> bitwise identical values
        x_r = tl.load(a_ptr + tok_r * stride_a + hv, mask=rmask, other=0.0).to(tl.float32) + dt_b
        x_c = tl.load(a_ptr + tok_c * stride_a + hv, mask=cmask, other=0.0).to(tl.float32) + dt_b
        sp_r = tl.where(sp_beta * x_r <= sp_threshold,
                        (1 / sp_beta) * tl.log(1 + tl.exp(sp_beta * x_r)), x_r)
        sp_c = tl.where(sp_beta * x_c <= sp_threshold,
                        (1 / sp_beta) * tl.log(1 + tl.exp(sp_beta * x_c)), x_c)
        g_r = tl.where(rmask, -A * sp_r, 0.0)
        g_c = tl.where(cmask, -A * sp_c, 0.0)
        beta_r = tl.where(rmask, tl.sigmoid(tl.load(b_ptr + tok_r * stride_b + hv, mask=rmask,
                                                    other=0.0).to(tl.float32)), 0.0)
        beta_c = tl.where(cmask, tl.sigmoid(tl.load(b_ptr + tok_c * stride_b + hv, mask=cmask,
                                                    other=0.0).to(tl.float32)), 0.0)

        # ancestor-or-self masks: M[i, j] = j anc-or-self of i; MT = M^T
        anc_r = tl.load(anc_ptr + n * stride_anc + ri, mask=rmask, other=0)
        anc_c = tl.load(anc_ptr + n * stride_anc + cj, mask=cmask, other=0)
        M = ((anc_r[:, None] >> cj[None, :].to(tl.int64)) & 1) != 0
        MT = ((anc_c[None, :] >> ri[:, None].to(tl.int64)) & 1) != 0
        G_r = tl.sum(tl.where(M, g_c[None, :], 0.0), axis=1)   # G_i (rows)
        G_c = tl.sum(tl.where(MT, g_r[:, None], 0.0), axis=0)  # G_j (cols)
        diag = ri[:, None] == cj[None, :]
        ratio = tl.where(M, tl.exp(tl.where(M, G_r[:, None] - G_c[None, :], 0.0)), 0.0)
        ratio = tl.where(diag & M, 1.0, ratio)
        strict = M & (ri[:, None] != cj[None, :])

        if BF16X3:
            # Exact-product tensor-core path: raw q/k are bf16 (exact), the
            # l2 norms are applied as fp32 row/col scales afterwards, and the
            # fp32 state is split into three bf16 pieces (h1+h2+h3 == h to
            # ~2^-24): every product is exact, accumulation is fp32 -- the
            # same error class as an IEEE fp32 dot, at MMA speed.
            rk_c = tl.sum(tl.where(diag, rk[:, None], 0.0), axis=0)
            rq_c = tl.sum(tl.where(diag, rq[:, None], 0.0), axis=0) * scale
            kb = k_raw.to(tl.bfloat16)
            qb = q_raw.to(tl.bfloat16)
            kbT = tl.trans(kb)
            qbT = tl.trans(qb)
            KK = tl.dot(kb, kbT) * (rk[:, None] * rk_c[None, :])
            QK = tl.dot(qb, kbT) * ((rq * scale)[:, None] * rk_c[None, :])
        else:
            kT = tl.trans(k)  # [K, NP]
            KK = tl.dot(k, kT, input_precision=IP)  # [NP, NP]
            QK = tl.dot(q, kT, input_precision=IP)

        # ---- 3) P = (I + A)^-1 by repeated squaring (A strictly lower, nilpotent)
        X = -(tl.where(strict, ratio * KK, 0.0) * beta_r[:, None])
        P = tl.where(diag, 1.0, 0.0) + X
        Y = X
        for _ in tl.static_range(LOG2NP - 1):
            Y = tl.dot(Y, Y, input_precision=IP)
            P = P + tl.dot(P, Y, input_precision=IP)

        # ---- 4) state-side products in transposed form ([BV, NP] tiles)
        if BF16X3:
            h1 = h.to(tl.bfloat16)
            r1 = h - h1.to(tl.float32)
            h2 = r1.to(tl.bfloat16)
            h3 = (r1 - h2.to(tl.float32)).to(tl.bfloat16)
            KSt = tl.dot(h3, kbT)
            KSt = tl.dot(h2, kbT, KSt)
            KSt = tl.dot(h1, kbT, KSt) * rk_c[None, :]
            QSt = tl.dot(h3, qbT)
            QSt = tl.dot(h2, qbT, QSt)
            QSt = tl.dot(h1, qbT, QSt) * rq_c[None, :]
        else:
            KSt = tl.dot(h, kT, input_precision=IP_S)  # (S0 k_j) rows v
            QSt = tl.dot(h, tl.trans(q), input_precision=IP_S)
        eG_c = tl.exp(G_c)
        rhst = beta_c[None, :] * (vt - eG_c[None, :] * KSt)
        Wt = tl.dot(rhst, tl.trans(P), input_precision=IP)  # W^T = R^T P^T
        Pq = tl.where(M, ratio * QK, 0.0)
        ot = eG_c[None, :] * QSt + tl.dot(Wt, tl.trans(Pq), input_precision=IP)
        tl.store(o_ptrs, ot.to(o_ptr.dtype.element_ty), mask=cmask[None, :])

        # ---- 5) commit: state after the root -> slot 0 (the single write)
        sel0c = cj == 0
        w0 = tl.sum(tl.where(sel0c[None, :], Wt, 0.0), axis=1)  # [BV]
        k0 = tl.sum(tl.where((ri == 0)[:, None], k, 0.0), axis=0)  # [K]
        g0 = tl.sum(tl.where(sel0c, g_c, 0.0), axis=0)
        h_root = h * tl.exp(g0) + w0[:, None] * k0[None, :]
        tl.store(h_ptrs, h_root.to(state_ptr.dtype.element_ty))

        # ---- 6) log nodes 1..T-1 (entry = node - 1) into the other buffer
        lmask_r = rmask & (ri >= 1)
        lmask_c = cmask & (cj >= 1)
        if (iv == 0) & (hv % G == 0):
            tl.store(wbuf + ((ri - 1)[:, None] * H + hk) * K + offs_k[None, :], k,
                     mask=lmask_r[:, None])
        tl.store(wbuf + V_OFF + ((cj - 1)[None, :] * HV + hv) * V + offs_v[:, None], vt,
                 mask=lmask_c[None, :])
        if iv == 0:
            tl.store(wbuf + D_OFF + (cj - 1) * HV + hv, tl.exp(g_c), mask=lmask_c)
            tl.store(wbuf + B_OFF + (cj - 1) * HV + hv, beta_c, mask=lmask_c)
        tl.store(par_addr, 1 - p)

        # ---- 7) P1-F1: gated RMSNorm of the head's output by the last of the
        # NV programs of (n, hv) (ticket), replacing the separate FLA
        # layer_norm_fwd launch and the strided-gate copy. Same fp32 op order
        # as layernorm_guard.layer_norm_fwd_kernel (norm_before_gate, rms).
        if FUSE_NORM:
            # __threadfence(); __syncthreads(); ticket (acq_rel) -- as marlin_plus
            dummy = tl.zeros([1], dtype=tl.int32)
            tl.inline_asm_elementwise("membar.gl; mov.b32 $0, $1;", "=r,r", [dummy],
                                      dtype=tl.int32, is_pure=False, pack=1)
            tl.debug_barrier()
            t = tl.atomic_add(tick_ptr + n * HV + hv, 1, sem="acq_rel")
            if t == NV - 1:
                offs_all = tl.arange(0, V)
                po = o_ptr + tok_c[:, None] * stride_o + hv * V + offs_all[None, :]
                xo = tl.load(po, mask=cmask[:, None], other=0.0).to(tl.float32)  # [NP, V]
                var = tl.sum(xo * xo, axis=1) / V
                rstd = tl.rsqrt(var + eps)
                w = tl.load(nw_ptr + offs_all).to(tl.float32)
                y = (xo * rstd[:, None]) * w[None, :]
                z = tl.load(z_ptr + tok_c[:, None] * stride_z + hv * V + offs_all[None, :],
                            mask=cmask[:, None], other=0.0).to(tl.float32)
                if SIGMOID:
                    y *= tl.sigmoid(z)
                else:
                    y *= z * tl.sigmoid(z)
                tl.store(po, y.to(o_ptr.dtype.element_ty), mask=cmask[:, None])
                tl.atomic_xchg(tick_ptr + n * HV + hv, 0)

    @triton.jit
    def _conv_tree_kernel(
        x_ptr, stride_x, w_ptr, bias_ptr,
        cs_ptr, stride_cs_slot, stride_cs_dim, stride_cs_tok,
        si_ptr, stride_si, cu_ptr, acc_ptr,
        src_ptr, stride_src_req,
        o_ptr, stride_o, dim,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr, BC: tl.constexpr,
        STATE_LEN: tl.constexpr,
    ):
        """Stock-layout conv state [h2, h3, x_0..x_{T-1}], window at offset
        a-1 (exactly stock's spec-decode convention); node t convolves with
        conv_src[t] = (1st, 2nd, 3rd ancestor), negatives = window (-1 newest)."""
        n = tl.program_id(0)
        cb = tl.program_id(1)
        bos = tl.load(cu_ptr + n).to(tl.int64)
        eos = tl.load(cu_ptr + n + 1).to(tl.int64)
        T = eos - bos
        if T <= 0:
            return
        slot = tl.load(si_ptr + n * stride_si).to(tl.int64)
        if slot <= 0:
            return
        ch = cb * BC + tl.arange(0, BC)
        cm = ch < dim
        base = cs_ptr + slot * stride_cs_slot + ch * stride_cs_dim
        off = tl.load(acc_ptr + n).to(tl.int64) - 1
        w_0 = tl.load(base + off * stride_cs_tok, mask=cm, other=0.0)
        w_1 = tl.load(base + (off + 1) * stride_cs_tok, mask=cm, other=0.0)
        w_2 = tl.load(base + (off + 2) * stride_cs_tok, mask=cm, other=0.0)
        wt0 = tl.load(w_ptr + ch * 4 + 0, mask=cm, other=0.0).to(tl.float32)
        wt1 = tl.load(w_ptr + ch * 4 + 1, mask=cm, other=0.0).to(tl.float32)
        wt2 = tl.load(w_ptr + ch * 4 + 2, mask=cm, other=0.0).to(tl.float32)
        wt3 = tl.load(w_ptr + ch * 4 + 3, mask=cm, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + ch, mask=cm, other=0.0).to(tl.float32)
        else:
            bias = tl.zeros([BC], dtype=tl.float32)
        tl.debug_barrier()
        f0 = w_0.to(tl.float32)
        f1 = w_1.to(tl.float32)
        f2 = w_2.to(tl.float32)
        for t in range(0, T):
            xt = tl.load(x_ptr + (bos + t) * stride_x + ch, mask=cm, other=0.0).to(tl.float32)
            y = bias + wt3 * xt
            for s in tl.static_range(3):
                j = tl.load(src_ptr + n * stride_src_req + t * 3 + s)
                jj = tl.maximum(j, 0)
                xa = tl.load(x_ptr + (bos + jj) * stride_x + ch, mask=cm, other=0.0).to(tl.float32)
                wa = tl.where(j == -1, f2, tl.where(j == -2, f1, f0))
                col = tl.where(j >= 0, xa, wa)
                if s == 0:
                    y += wt2 * col
                elif s == 1:
                    y += wt1 * col
                else:
                    y += wt0 * col
            if ACT:
                y = y * tl.sigmoid(y)
            tl.store(o_ptr + (bos + t) * stride_o + ch, y.to(o_ptr.dtype.element_ty), mask=cm)
        tl.store(base + 0 * stride_cs_tok, w_1, mask=cm)
        tl.store(base + 1 * stride_cs_tok, w_2, mask=cm)
        for t in range(0, T):
            xt = tl.load(x_ptr + (bos + t) * stride_x + ch, mask=cm, other=0.0)
            tl.store(base + (2 + t) * stride_cs_tok, xt.to(cs_ptr.dtype.element_ty),
                     mask=cm & (2 + t < STATE_LEN))

    @triton.jit
    def _conv_tree_vec_kernel(
        x_ptr, stride_x, w_ptr, bias_ptr,
        cs_ptr, stride_cs_slot, stride_cs_dim, stride_cs_tok,
        si_ptr, stride_si, cu_ptr, acc_ptr,
        src_ptr, stride_src_req,
        o_ptr, stride_o, dim,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr, BC: tl.constexpr,
        STATE_LEN: tl.constexpr, TP: tl.constexpr,
    ):
        """Same math (and fp32 op order) as _conv_tree_kernel, but all T <= TP
        tokens of a channel block are computed as one [TP, BC] tile, so the
        token / ancestor loads are issued together instead of a serial,
        latency-bound loop over T (reframe B: 12 us -> few us per layer)."""
        n = tl.program_id(0)
        cb = tl.program_id(1)
        bos = tl.load(cu_ptr + n).to(tl.int64)
        eos = tl.load(cu_ptr + n + 1).to(tl.int64)
        T = eos - bos
        if T <= 0:
            return
        slot = tl.load(si_ptr + n * stride_si).to(tl.int64)
        if slot <= 0:
            return
        ch = cb * BC + tl.arange(0, BC)
        cm = ch < dim
        tt = tl.arange(0, TP)
        tm = tt < T
        m2 = tm[:, None] & cm[None, :]
        base = cs_ptr + slot * stride_cs_slot + ch * stride_cs_dim
        off = tl.load(acc_ptr + n).to(tl.int64) - 1
        w_0 = tl.load(base + off * stride_cs_tok, mask=cm, other=0.0)
        w_1 = tl.load(base + (off + 1) * stride_cs_tok, mask=cm, other=0.0)
        w_2 = tl.load(base + (off + 2) * stride_cs_tok, mask=cm, other=0.0)
        wt0 = tl.load(w_ptr + ch * 4 + 0, mask=cm, other=0.0).to(tl.float32)
        wt1 = tl.load(w_ptr + ch * 4 + 1, mask=cm, other=0.0).to(tl.float32)
        wt2 = tl.load(w_ptr + ch * 4 + 2, mask=cm, other=0.0).to(tl.float32)
        wt3 = tl.load(w_ptr + ch * 4 + 3, mask=cm, other=0.0).to(tl.float32)
        if HAS_BIAS:
            bias = tl.load(bias_ptr + ch, mask=cm, other=0.0).to(tl.float32)
        else:
            bias = tl.zeros([BC], dtype=tl.float32)
        f0 = w_0.to(tl.float32)[None, :]
        f1 = w_1.to(tl.float32)[None, :]
        f2 = w_2.to(tl.float32)[None, :]
        xraw = tl.load(x_ptr + (bos + tt)[:, None] * stride_x + ch[None, :], mask=m2, other=0.0)
        xt = xraw.to(tl.float32)
        y = bias[None, :] + wt3[None, :] * xt
        for s in tl.static_range(3):
            j = tl.load(src_ptr + n * stride_src_req + tt * 3 + s, mask=tm, other=0)
            jj = tl.maximum(j, 0).to(tl.int64)
            xa = tl.load(x_ptr + (bos + jj)[:, None] * stride_x + ch[None, :], mask=m2,
                         other=0.0).to(tl.float32)
            jb = j[:, None]
            wa = tl.where(jb == -1, f2, tl.where(jb == -2, f1, f0))
            col = tl.where(jb >= 0, xa, wa)
            if s == 0:
                y += wt2[None, :] * col
            elif s == 1:
                y += wt1[None, :] * col
            else:
                y += wt0[None, :] * col
        if ACT:
            y = y * tl.sigmoid(y)
        tl.store(o_ptr + (bos + tt)[:, None] * stride_o + ch[None, :],
                 y.to(o_ptr.dtype.element_ty), mask=m2)
        tl.debug_barrier()  # every thread's window reads precede the state rewrite
        tl.store(base + 0 * stride_cs_tok, w_1, mask=cm)
        tl.store(base + 1 * stride_cs_tok, w_2, mask=cm)
        tl.store(base[None, :] + (2 + tt)[:, None] * stride_cs_tok,
                 xraw.to(cs_ptr.dtype.element_ty), mask=m2 & ((2 + tt) < STATE_LEN)[:, None])

    @triton.jit
    def _tree_accept_kernel(
        parent_ptr, stride_par, tokens_ptr, stride_tok, argmax_ptr, stride_am,
        path_ptr, sampled_ptr, nsamp_ptr, stride_out, N,
        NP: tl.constexpr, MAX_DEPTH: tl.constexpr, OUT_COLS: tl.constexpr,
    ):
        """Greedy tree acceptance for one request: walk the target argmax.
        path[0..a-1] = accepted node indices (path[0] = 0); sampled[0..a-1] =
        their tokens (excluding the root) followed by the bonus token; -1 pad."""
        r = tl.program_id(0)
        rows = tl.arange(0, NP)
        valid = rows < N
        par = tl.load(parent_ptr + r * stride_par + rows, mask=valid, other=-2)
        toks = tl.load(tokens_ptr + r * stride_tok + rows, mask=valid, other=-1).to(tl.int64)
        cols = tl.arange(0, OUT_COLS)
        tl.store(path_ptr + r * stride_out + cols, -1)
        tl.store(sampled_ptr + r * stride_out + cols, -1)
        node = tl.full([], 0, tl.int32)
        alive = tl.full([], 1, tl.int32)
        cnt = tl.full([], 1, tl.int32)
        tl.store(path_ptr + r * stride_out, 0)
        for d in range(0, MAX_DEPTH):
            want = tl.load(argmax_ptr + r * stride_am + node).to(tl.int64)
            hit = (par == node) & (toks == want) & (rows > 0) & valid
            nxt = tl.max(tl.where(hit, rows, -1), axis=0).to(tl.int32)
            go = (alive != 0) & (nxt >= 0)
            tl.store(path_ptr + r * stride_out + cnt, nxt, mask=go)
            tl.store(sampled_ptr + r * stride_out + cnt - 1, want, mask=go)
            cnt = tl.where(go, cnt + 1, cnt)
            node = tl.where(go, nxt, node)
            alive = go.to(tl.int32)
        bonus = tl.load(argmax_ptr + r * stride_am + node).to(tl.int64)
        tl.store(sampled_ptr + r * stride_out + cnt - 1, bonus)
        tl.store(nsamp_ptr + r, cnt)

    @triton.jit
    def _tree_build_kernel(
        scores_ptr, cand_ptr, rmap_ptr, rmap_stride,
        tok_out_ptr, stride_tok_out,
        mpar_ptr, mdep_ptr, manc_ptr, mconv_ptr,
        inv_temp, unary_ptr, mix_a, pw,
        N: tl.constexpr, NP: tl.constexpr, K: tl.constexpr, D: tl.constexpr,
        CHAIN: tl.constexpr, MIX: tl.constexpr = False,
    ):
        """Best-first (k-best paths) tree over DFlash2's CRF lattice for one
        request (== tree_spec.build_tree with edge_logprobs(scores / T)), or,
        with CHAIN, vLLM's greedy walk (D nodes) padded with dead leaves.

        scores [R, D, K, K] fp32, cand [R, D, K] int64. Writes the N-1 draft
        tokens (batch row) and per-request-state metadata (parent, depth,
        ancestor bits, conv sources) at rmap[r] (skipped when < 0)."""
        r = tl.program_id(0)
        rs = tl.load(rmap_ptr + r * rmap_stride)
        nodes = tl.arange(0, NP)  # NP = next_pow2(N); rows >= N stay dead
        vm = nodes < N
        kk = tl.arange(0, K)
        sbase = scores_ptr + r * (D * K * K)
        cbase = cand_ptr + r * (D * K)
        NEG = float("-inf")
        val = tl.where(nodes == 0, 0.0, NEG)
        dep = tl.zeros([NP], dtype=tl.int32)
        cid = tl.zeros([NP], dtype=tl.int32)
        par = tl.where(nodes == 0, -1, 0)
        tok = tl.zeros([NP], dtype=tl.int64)
        anc = tl.where(nodes == 0, 1, 0).to(tl.int64)
        used = tl.zeros([NP], dtype=tl.int32)
        for it in range(1, N):
            alive = (nodes < it) & (dep < D)
            if CHAIN:
                alive = alive & (nodes == it - 1)
            rowoff = (dep * K + cid) * K
            S = tl.load(sbase + rowoff[:, None] + kk[None, :], mask=alive[:, None],
                        other=0.0)
            if MIX:
                # E7 marginal mix: edge = (1-a) log_softmax((u + pw (s - u)) / T) + a log_softmax(u)
                U = tl.load(unary_ptr + r * (D * K) + (dep * K)[:, None] + kk[None, :],
                            mask=alive[:, None], other=0.0)
                S = (U + pw * (S - U)) * inv_temp
                mu = tl.max(U, axis=1)
                lu = U - (mu + tl.log(tl.sum(tl.exp(U - mu[:, None]), axis=1)))[:, None]
            else:
                S = S * inv_temp
            m = tl.max(S, axis=1)
            lse = m + tl.log(tl.sum(tl.exp(S - m[:, None]), axis=1))
            E = S - lse[:, None]
            if MIX:
                E = (1.0 - mix_a) * E + mix_a * lu
            usedbit = ((used[:, None] >> kk[None, :]) & 1) != 0
            cv = tl.where(alive[:, None] & (~usedbit), val[:, None] + E, NEG)
            rowbest = tl.max(cv, axis=1)
            best = tl.max(rowbest, axis=0)
            jstar = tl.argmax(rowbest, axis=0).to(tl.int32)
            crow = tl.max(tl.where(nodes[:, None] == jstar, cv, NEG), axis=0)
            cstar = tl.argmax(crow, axis=0).to(tl.int32)
            ok = best > NEG
            pdep = tl.sum(tl.where(nodes == jstar, dep, 0), axis=0)
            panc = tl.sum(tl.where(nodes == jstar, anc, 0), axis=0)
            ctok = tl.load(cbase + tl.minimum(pdep, D - 1) * K + cstar)
            sel = nodes == it
            bit = tl.full([], 1, tl.int64) << it
            par = tl.where(sel, tl.where(ok, jstar, 0), par)
            dep = tl.where(sel, tl.where(ok, pdep + 1, 1), dep)
            cid = tl.where(sel, tl.where(ok, cstar, 0), cid)
            val = tl.where(sel, tl.where(ok, best, NEG), val)
            tok = tl.where(sel, tl.where(ok, ctok, 0), tok)
            anc = tl.where(sel, tl.where(ok, panc, 1) | bit, anc)
            jsel = tl.where(ok, jstar, -1)
            used = tl.where(nodes == jsel, used | (1 << cstar), used)
        if rs >= 0:
            tl.store(tok_out_ptr + r * stride_tok_out + (nodes - 1), tok,
                     mask=(nodes >= 1) & vm)
            mb = rs.to(tl.int64) * N
            tl.store(mpar_ptr + mb + nodes, par, mask=vm)
            tl.store(mdep_ptr + mb + nodes, dep, mask=vm)
            tl.store(manc_ptr + mb + nodes, anc, mask=vm)
            # conv sources: 1st..3rd ancestor; root -> -1, -2, -3 (window)
            c1 = par
            pg1 = tl.sum(tl.where(nodes[None, :] == c1[:, None], par[None, :], 0), axis=1)
            c2 = tl.where(c1 > 0, pg1, tl.where(c1 == 0, -1, c1 - 1))
            pg2 = tl.sum(tl.where(nodes[None, :] == c2[:, None], par[None, :], 0), axis=1)
            c3 = tl.where(c2 > 0, pg2, tl.where(c2 == 0, -1, c2 - 1))
            tl.store(mconv_ptr + (mb + nodes) * 3 + 0, c1, mask=vm)
            tl.store(mconv_ptr + (mb + nodes) * 3 + 1, c2, mask=vm)
            tl.store(mconv_ptr + (mb + nodes) * 3 + 2, c3, mask=vm)

    @triton.jit
    def _tree_activate_kernel(
        idx_ptr, mpar_ptr, mdep_ptr, manc_ptr, mconv_ptr,
        apar_ptr, adep_ptr, aanc_ptr, aconv_ptr, qq_ptr, idx_out_ptr,
        N: tl.constexpr, NP: tl.constexpr, TREE: tl.constexpr, STASH: tl.constexpr = False,
    ):
        """Row 0 of the active (batch-row) metadata <- the tree of request
        state idx[0] (TREE) or the causal chain; qq_bias [N, N] accordingly.
        STASH: also idx_out[0] = idx[0] (persistent copy for the accept graph)."""
        nodes = tl.arange(0, NP)
        vm = nodes < N
        if STASH:
            tl.store(idx_out_ptr, tl.load(idx_ptr))
        if TREE:
            rs = tl.load(idx_ptr).to(tl.int64)
            par = tl.load(mpar_ptr + rs * N + nodes, mask=vm, other=0)
            dep = tl.load(mdep_ptr + rs * N + nodes, mask=vm, other=0)
            anc = tl.load(manc_ptr + rs * N + nodes, mask=vm, other=0)
            c1 = tl.load(mconv_ptr + (rs * N + nodes) * 3 + 0, mask=vm, other=0)
            c2 = tl.load(mconv_ptr + (rs * N + nodes) * 3 + 1, mask=vm, other=0)
            c3 = tl.load(mconv_ptr + (rs * N + nodes) * 3 + 2, mask=vm, other=0)
        else:
            par = nodes - 1
            dep = nodes
            anc = (tl.full([NP], 2, tl.int64) << nodes.to(tl.int64)) - 1
            c1 = nodes - 1
            c2 = nodes - 2
            c3 = nodes - 3
        tl.store(apar_ptr + nodes, par, mask=vm)
        tl.store(adep_ptr + nodes, dep, mask=vm)
        tl.store(aanc_ptr + nodes, anc, mask=vm)
        tl.store(aconv_ptr + nodes * 3 + 0, c1, mask=vm)
        tl.store(aconv_ptr + nodes * 3 + 1, c2, mask=vm)
        tl.store(aconv_ptr + nodes * 3 + 2, c3, mask=vm)
        if TREE:
            bits = ((anc[:, None] >> nodes[None, :].to(tl.int64)) & 1) != 0
            qq = tl.where(bits, 0.0, float("-inf"))
        else:
            qq = tl.zeros([NP, NP], dtype=tl.float32)
        tl.store(qq_ptr + nodes[:, None] * N + nodes[None, :], qq,
                 mask=vm[:, None] & vm[None, :])

    @triton.jit
    def _tree_pos_kernel(pos_ptr, row_stride, start_ptr, dep_ptr,
                         N: tl.constexpr, NP: tl.constexpr, NROWS: tl.constexpr,
                         SIGN: tl.constexpr):
        """positions[row, start + i] += SIGN * (depth[i] - i)."""
        nodes = tl.arange(0, NP)
        vm = nodes < N
        st = tl.load(start_ptr).to(tl.int64)
        dlt = (tl.load(dep_ptr + nodes, mask=vm, other=0) - nodes).to(tl.int64) * SIGN
        for rr in tl.static_range(NROWS):
            ptr = pos_ptr + rr * row_stride + st + nodes
            x = tl.load(ptr, mask=vm, other=0)
            tl.store(ptr, (x + dlt).to(pos_ptr.dtype.element_ty), mask=vm)

    @triton.jit
    def _compact_log_kernel(
        ptr_tab, slot_tab, grp_tab, si_ptr, si_sg, si_sr,
        path_ptr, stride_path, ns_ptr,
        LOG_BASE: tl.constexpr, BUF: tl.constexpr, HK: tl.constexpr,
        HVV: tl.constexpr, HV: tl.constexpr, V_OFF: tl.constexpr,
        D_OFF: tl.constexpr, B_OFF: tl.constexpr, BLOCK: tl.constexpr,
        MAXD: tl.constexpr,
    ):
        """Move log entries path[d]-1 -> d-1 (d = 1..a-1) in the buffer the
        last tree step wrote (= current parity), for every GDN layer."""
        layer = tl.program_id(0)
        r = tl.program_id(1)
        cb = tl.program_id(2)
        a = tl.load(ns_ptr + r)
        if a <= 1:
            return
        g = tl.load(grp_tab + layer)
        s1 = tl.load(si_ptr + g * si_sg + r * si_sr + 1).to(tl.int64)
        if s1 <= 0:
            return
        base = tl.load(ptr_tab + layer).to(tl.pointer_type(tl.float32))
        sb = base + s1 * tl.load(slot_tab + layer)
        p = tl.load(sb.to(tl.pointer_type(tl.int32), bitcast=True)) & 1
        buf = sb + LOG_BASE + p * BUF
        c = cb * BLOCK + tl.arange(0, BLOCK)
        E: tl.constexpr = HK + HVV + 2 * HV
        cm = c < E
        for d in tl.static_range(1, MAXD):
            if d < a:
                src = (tl.load(path_ptr + r * stride_path + d) - 1).to(tl.int64)
                dst = d - 1
                if src != dst:
                    o_s = tl.where(c < HK, src * HK + c,
                                   tl.where(c < HK + HVV, V_OFF + src * HVV + (c - HK),
                                            tl.where(c < HK + HVV + HV,
                                                     D_OFF + src * HV + (c - HK - HVV),
                                                     B_OFF + src * HV + (c - HK - HVV - HV))))
                    o_d = tl.where(c < HK, dst * HK + c,
                                   tl.where(c < HK + HVV, V_OFF + dst * HVV + (c - HK),
                                            tl.where(c < HK + HVV + HV,
                                                     D_OFF + dst * HV + (c - HK - HVV),
                                                     B_OFF + dst * HV + (c - HK - HVV - HV))))
                    x = tl.load(buf + o_s, mask=cm)
                    tl.store(buf + o_d, x, mask=cm)

    @triton.jit
    def _compact_conv_kernel(
        ptr_tab, grp_tab, si_ptr, si_sg, si_sr, path_ptr, stride_path, ns_ptr,
        s_slot, s_dim, s_tok, dim,
        ELEM: tl.constexpr, BLOCK: tl.constexpr, MAXD: tl.constexpr,
    ):
        """Conv state column 2+path[d] -> 2+d (d = 1..a-1), every GDN layer."""
        layer = tl.program_id(0)
        r = tl.program_id(1)
        cb = tl.program_id(2)
        a = tl.load(ns_ptr + r)
        if a <= 1:
            return
        g = tl.load(grp_tab + layer)
        s0 = tl.load(si_ptr + g * si_sg + r * si_sr).to(tl.int64)
        if s0 <= 0:
            return
        if ELEM == 2:
            base = tl.load(ptr_tab + layer).to(tl.pointer_type(tl.int16))
        else:
            base = tl.load(ptr_tab + layer).to(tl.pointer_type(tl.int32))
        ch = cb * BLOCK + tl.arange(0, BLOCK)
        cm = ch < dim
        sb = base + s0 * s_slot + ch * s_dim
        for d in tl.static_range(1, MAXD):
            if d < a:
                src = tl.load(path_ptr + r * stride_path + d).to(tl.int64)
                if src != d:
                    x = tl.load(sb + (2 + src) * s_tok, mask=cm)
                    tl.store(sb + (2 + d) * s_tok, x, mask=cm)

    @triton.jit
    def _compact_kv_kernel(
        ptr_tab, grp_tab, sm_ptr, sm_stride, qsl_ptr, path_ptr, stride_path, ns_ptr,
        block_size, s_blk, s_head, s_off,
        HEADS: tl.constexpr, E: tl.constexpr, ELEM: tl.constexpr,
        BLOCK: tl.constexpr, MAXD: tl.constexpr,
    ):
        """Paged K/V of node path[d] (slot of row qs+path[d]) -> slot of row
        qs+d, for every full-attention layer. kv[blk, head, off, e]."""
        layer = tl.program_id(0)
        r = tl.program_id(1)
        cb = tl.program_id(2)
        a = tl.load(ns_ptr + r)
        if a <= 1:
            return
        if ELEM == 2:
            base = tl.load(ptr_tab + layer).to(tl.pointer_type(tl.int16))
        else:
            base = tl.load(ptr_tab + layer).to(tl.pointer_type(tl.int8))
        qs = tl.load(qsl_ptr + r).to(tl.int64)
        sm_ptr = sm_ptr + tl.load(grp_tab + layer).to(tl.int64) * sm_stride
        c = cb * BLOCK + tl.arange(0, BLOCK)
        cm = c < HEADS * E
        hd = c // E
        ee = c % E
        for d in tl.static_range(1, MAXD):
            if d < a:
                src = tl.load(path_ptr + r * stride_path + d).to(tl.int64)
                if src != d:
                    ss = tl.load(sm_ptr + qs + src).to(tl.int64)
                    ds = tl.load(sm_ptr + qs + d).to(tl.int64)
                    if (ss >= 0) & (ds >= 0):
                        so = (ss // block_size) * s_blk + (ss % block_size) * s_off
                        do = (ds // block_size) * s_blk + (ds % block_size) * s_off
                        x = tl.load(base + so + hd * s_head + ee, mask=cm)
                        tl.store(base + do + hd * s_head + ee, x, mask=cm)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def _rows2d(x: torch.Tensor, width: int) -> torch.Tensor:
    x2 = x.reshape(x.shape[0], -1) if x.dim() > 2 else x
    assert x2.shape[1] == width and x2.stride(1) == 1, (x2.shape, x2.stride())
    return x2


def tree_log_layout(state: torch.Tensor, num_k_heads: int, num_slots: int, block_v: int):
    """A3 log layout for a spec slot row of ``num_slots`` (max log = S-1)."""
    _, hv, vdim, kdim = state.shape
    lay = log_layout(num_k_heads, hv, kdim, vdim, block_v, num_slots - 1)
    if lay["total"] > slot_capacity_f32(state):
        raise ValueError(f"gdn_tree: log for {num_slots} nodes needs {lay['total']} fp32 "
                         f"per slot, slot has {slot_capacity_f32(state)}")
    return lay


_TICK: dict = {}
TICK_SLOTS = 8192


def _tick(dev: torch.device) -> torch.Tensor:
    t = _TICK.get(dev)
    if t is None:
        t = _TICK[dev] = torch.zeros(TICK_SLOTS, dtype=torch.int32, device=dev)
    return t


def gdn_tree_verify(*, q, k, v, a, b, A_log, dt_bias, state, state_indices,
                    cu_seqlens, num_accepted, anc_bits, out, num_k_heads, scale,
                    block_v=32, input_precision="ieee", state_precision="bf16x3",
                    softplus_beta=1.0, softplus_threshold=20.0,
                    num_requests=None, output_gate=None, norm_weight=None, norm_eps=1e-6,
                    sigmoid_gate=False) -> None:
    """Tree verify, A3 slot layout. state_indices [R, S] (S = node budget;
    log holds S-1 entries). anc_bits [R, >= T] int64 (bit j of row i = node j
    is an ancestor-or-self of node i). Output: un-normed o [L, HV, V].
    state_precision: "bf16x3" (default; exact bf16 products of a 3-way split
    fp32 state on tensor cores, fp32 accumulate; needs bf16 q/k, else falls
    back to "ieee"), "ieee" (FMA), "tf32x3", "tf32" (timing only).
    output_gate [L, HV, V] (any row stride, head stride V) + norm_weight [V]:
    the gated RMSNorm is applied in-kernel (P1-F1) and `out` holds the normed
    output; otherwise `out` is un-normed and the caller norms it."""
    slots, hv, vdim, kdim = state.shape
    h = num_k_heads
    assert state.dtype in (torch.float32, torch.bfloat16)
    assert state.stride()[1:] == (vdim * kdim, kdim, 1), state.stride()
    assert hv % h == 0 and vdim % block_v == 0 and block_v >= 16
    S = state_indices.size(1)
    np_ = max(16, triton.next_power_of_2(S))
    assert np_ <= 64
    lay = tree_log_layout(state, h, S, block_v)
    n = state_indices.size(0) if num_requests is None else num_requests
    if n == 0:
        return
    q2, k2, v2 = _rows2d(q, h * kdim), _rows2d(k, h * kdim), _rows2d(v, hv * vdim)
    assert out.stride(-1) == 1 and out.stride(-2) == vdim
    log_view = state.view(torch.float32)
    par_view = state.view(torch.int32)
    fuse = output_gate is not None
    if fuse:
        assert norm_weight is not None and n * hv <= TICK_SLOTS
        assert output_gate.dim() == 3 and output_gate.stride(2) == 1 \
            and output_gate.stride(1) == vdim and output_gate.shape[1] == hv
        z, stride_z, nw = output_gate, output_gate.stride(0), norm_weight
        tick = _tick(out.device)
    else:
        z, stride_z, nw, tick = out, 0, out, out
    _gdn_tree_kernel[(n, hv, vdim // block_v)](
        q2, k2, v2, q2.stride(0), k2.stride(0), v2.stride(0),
        a, b, a.stride(0), b.stride(0), A_log, dt_bias,
        state, log_view, par_view,
        state.stride(0), log_view.stride(0), par_view.stride(0),
        state_indices, state_indices.stride(0), cu_seqlens, num_accepted,
        anc_bits, anc_bits.stride(0),
        out, out.stride(0),
        z, stride_z, nw, tick, float(norm_eps),
        float(scale), float(softplus_beta), float(softplus_threshold),
        H=h, HV=hv, K=kdim, V=vdim, BV=block_v, NP=np_,
        LOG2NP=np_.bit_length() - 1, MAX_LOG=S - 1,
        LOG_BASE=lay["log_base"], BUF=lay["buf"],
        V_OFF=lay["v_off"], D_OFF=lay["d_off"], B_OFF=lay["b_off"],
        IP=input_precision,
        IP_S="ieee" if state_precision == "bf16x3" else state_precision,
        BF16X3=state_precision == "bf16x3" and q.dtype == torch.bfloat16
        and k.dtype == torch.bfloat16,
        FUSE_NORM=fuse, SIGMOID=bool(sigmoid_gate),
        num_warps=4 if block_v <= 32 else 8,
    )


def conv_tree_update(*, x, weight, bias, conv_state, state_indices, cu_seqlens,
                     num_accepted, conv_src, out, activation=True, block_c=None,
                     num_requests=None) -> None:
    """x [L, C] (row stride any); weight [C, 4]; conv_state [slots, C,
    state_len] (any strides, stock spec layout); conv_src [R, >=T, 3] int32."""
    n = state_indices.size(0) if num_requests is None else num_requests
    if n == 0:
        return
    dim = x.shape[1]
    assert x.stride(1) == 1 and out.stride(1) == 1 and weight.is_contiguous()
    assert conv_src.stride(2) == 1 and conv_src.stride(1) == 3
    if block_c is None:  # reframe B: more CTAs for the latency-bound c=1 case
        block_c = int(os.environ.get("ATDEV_TREE_CONV_BC", "128"))
    common = (x, x.stride(0), weight, bias if bias is not None else weight,
              conv_state, conv_state.stride(0), conv_state.stride(1), conv_state.stride(2),
              state_indices, state_indices.stride(0), cu_seqlens, num_accepted,
              conv_src, conv_src.stride(0), out, out.stride(0), dim)
    tp = conv_src.shape[1]
    if os.environ.get("ATDEV_TREE_CONV_VEC", "1") == "1" and tp <= 32:
        # conv_src has a row per node (N); every request's T <= N
        bc = int(os.environ.get("ATDEV_TREE_CONV_VEC_BC", "32"))
        _conv_tree_vec_kernel[(n, triton.cdiv(dim, bc))](
            *common, HAS_BIAS=bias is not None, ACT=activation, BC=bc,
            STATE_LEN=conv_state.shape[2], TP=triton.next_power_of_2(tp),
            num_warps=4)
        return
    _conv_tree_kernel[(n, triton.cdiv(dim, block_c))](
        *common, HAS_BIAS=bias is not None, ACT=activation, BC=block_c,
        STATE_LEN=conv_state.shape[2], num_warps=max(1, min(4, block_c // 32)),
    )


def tree_accept(parent, tokens, target_argmax, max_depth, out_cols=None,
                path=None, sampled=None, nsamp=None):
    """parent/tokens/target_argmax [R, N] (any int dtypes, row strides any)
    -> (path [R, C] int32, sampled [R, C] int64, num_sampled [R] int32),
    C = out_cols (>= max_depth + 1), -1 padded."""
    r, n = parent.shape
    cols = max_depth + 1 if out_cols is None else out_cols
    assert cols >= max_depth + 1
    cp = triton.next_power_of_2(cols)
    dev = parent.device
    if path is None:
        path = torch.empty((r, cp), dtype=torch.int32, device=dev)
    if sampled is None:
        sampled = torch.empty((r, cp), dtype=torch.int64, device=dev)
    if nsamp is None:
        nsamp = torch.empty(r, dtype=torch.int32, device=dev)
    assert path.stride(0) == sampled.stride(0)
    _tree_accept_kernel[(r,)](
        parent, parent.stride(0), tokens, tokens.stride(0),
        target_argmax, target_argmax.stride(0),
        path, sampled, nsamp, path.stride(0), n,
        NP=triton.next_power_of_2(n), MAX_DEPTH=max_depth, OUT_COLS=cp,
        num_warps=1,
    )
    return path[:, :cols], sampled[:, :cols], nsamp


def build_tree_gpu(scores, cand, rmap, rmap_stride, tok_out, meta_parent, meta_depth,
                   meta_anc, meta_conv, num_nodes, chain=False, temperature=1.0,
                   unary=None, mix=0.0, pw=1.0):
    """scores [R, D, K, K] fp32 contiguous, cand [R, D, K] int64 contiguous.
    unary [R, D, K] fp32 (E7 marginal mix, used when mix > 0 or pw != 1)."""
    r, d, kc, _ = scores.shape
    assert scores.is_contiguous() and cand.is_contiguous()
    assert 2 <= num_nodes <= 63 and kc & (kc - 1) == 0
    use_mix = unary is not None and (mix != 0.0 or pw != 1.0)
    if use_mix:
        assert unary.is_contiguous() and unary.dtype == torch.float32 and tuple(unary.shape) == (r, d, kc)
    _tree_build_kernel[(r,)](
        scores, cand, rmap, rmap_stride, tok_out, tok_out.stride(0),
        meta_parent, meta_depth, meta_anc, meta_conv, 1.0 / float(temperature),
        unary if use_mix else scores, float(mix), float(pw),
        N=num_nodes, NP=triton.next_power_of_2(num_nodes), K=kc, D=d, CHAIN=chain, MIX=use_mix,
        num_warps=1,
    )


def tree_activate(idx_mapping, meta, act, qq, num_nodes, tree: bool, idx_out=None):
    """idx_out: optional persistent buffer that receives idx_mapping[0] (same dtype)."""
    if idx_out is not None:
        assert idx_out.dtype == idx_mapping.dtype
    _tree_activate_kernel[(1,)](
        idx_mapping, meta["parent"], meta["depth"], meta["anc"], meta["conv"],
        act["parent"], act["depth"], act["anc"], act["conv"], qq,
        idx_mapping if idx_out is None else idx_out,
        N=num_nodes, NP=triton.next_power_of_2(num_nodes), TREE=tree,
        STASH=idx_out is not None, num_warps=1,
    )


def tree_shift_positions(pos, start, depth, num_nodes, sign):
    """pos [T] or [R, T] (rows share the offset); start: 1-elem int tensor."""
    nrows = 1 if pos.dim() == 1 else pos.shape[0]
    rs = 0 if pos.dim() == 1 else pos.stride(0)
    assert pos.stride(-1) == 1
    _tree_pos_kernel[(1,)](pos, rs, start, depth, N=num_nodes,
                           NP=triton.next_power_of_2(num_nodes), NROWS=nrows,
                           SIGN=sign, num_warps=1)


def compact_gdn_log(*, ptr_tab, slot_tab, grp_tab, state_idx, path, num_sampled, lay,
                    num_k_heads, head_k_dim, num_v_heads, head_v_dim, max_depth,
                    num_layers, num_reqs, block=1024):
    """state_idx [G, R, >=2] (per kv-cache group spec slots); path [R, >=D+1]."""
    hk = num_k_heads * head_k_dim
    hvv = num_v_heads * head_v_dim
    ent = hk + hvv + 2 * num_v_heads
    _compact_log_kernel[(num_layers, num_reqs, triton.cdiv(ent, block))](
        ptr_tab, slot_tab, grp_tab, state_idx, state_idx.stride(0), state_idx.stride(1),
        path, path.stride(0), num_sampled,
        LOG_BASE=lay["log_base"], BUF=lay["buf"], HK=hk, HVV=hvv, HV=num_v_heads,
        V_OFF=lay["v_off"], D_OFF=lay["d_off"], B_OFF=lay["b_off"], BLOCK=block,
        MAXD=max_depth + 1, num_warps=4,
    )


def compact_conv_state(*, ptr_tab, grp_tab, state_idx, path, num_sampled, strides,
                       dim, elem_size, max_depth, num_layers, num_reqs, block=1024):
    """strides = (slot, dim, tok) of the [slots, dim, state_len] conv view."""
    _compact_conv_kernel[(num_layers, num_reqs, triton.cdiv(dim, block))](
        ptr_tab, grp_tab, state_idx, state_idx.stride(0), state_idx.stride(1),
        path, path.stride(0), num_sampled, strides[0], strides[1], strides[2], dim,
        ELEM=elem_size, BLOCK=block, MAXD=max_depth + 1, num_warps=4,
    )


def compact_kv(*, ptr_tab, grp_tab, slot_mappings, query_start_loc, path, num_sampled,
               block_size, strides, heads, entry, elem_size, max_depth, num_layers,
               num_reqs, block=1024):
    """kv[blk, head, off, e] with strides (blk, head, off); e contiguous.
    slot_mappings [G, T]; grp_tab[layer] = row of the layer's kv-cache group."""
    _compact_kv_kernel[(num_layers, num_reqs, triton.cdiv(heads * entry, block))](
        ptr_tab, grp_tab, slot_mappings, slot_mappings.stride(0), query_start_loc, path,
        path.stride(0), num_sampled,
        block_size, strides[0], strides[1], strides[2],
        HEADS=heads, E=entry, ELEM=elem_size, BLOCK=block, MAXD=max_depth + 1,
        num_warps=4,
    )
