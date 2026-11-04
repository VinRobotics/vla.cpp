"""Fake-quantize pi0.5's action-expert GEMM weights to 2 bits, in place in a copy of the GGUF.

Accuracy feasibility check for a W2A8 expert kernel: each expert weight row is split
into groups of G along K; each group is quantized to 4 uniform levels between a
clipped [lo, hi] (asymmetric, min/max shrunk by a per-group factor chosen to minimize
the squared error), then dequantized and written back in the tensor's own type, so
the unmodified loader and kernels run the 2-bit-valued weights.

usage: fakequant_expert_2bit.py src.gguf dst.gguf [--group 64] [--bits 2]
"""
import argparse, re, shutil, sys, time
import numpy as np

sys.path.insert(0, "/home/sand/Projects/vla.cpp-npu-up/build-ov/_deps/llama-src/gguf-py")
import gguf  # noqa: E402

SITES = re.compile(r"^aex\.blk\.\d+\.(attn_q|attn_k|attn_v|attn_o|ffn_gate|ffn_up|ffn_down)\.weight$")


def fake_quant_pow2(w, group, max_shift=4, row_clip=False):
    """Kernel-friendly 2-bit: symmetric codes c in {-3,-1,1,3} (2*q-3, q in 0..3) times a
    power-of-two group scale 2^e (e in 0..max_shift) times one float scale per output row.
    Each weight is then the exact integer c << e (|c << e| <= 48, fits int8), so a kernel can
    feed INT8 DPAS directly and apply only the per-row float scale at the end."""
    n, k = w.shape
    g = w.reshape(n, k // group, group).astype(np.float32)
    gmax = np.abs(g).max(axis=2, keepdims=True)                      # per group
    rmax = np.maximum(gmax.max(axis=1, keepdims=True), 1e-12)        # per row
    best, best_err = None, None
    for a in np.linspace(0.55, 1.0, 10):
        row_s = a * rmax / (3 << max_shift)                          # value of integer 1
        # per group: smallest exponent whose range covers the (clipped) group max
        need = np.maximum(a * gmax / (3 * row_s), 1e-12)
        e = np.clip(np.ceil(np.log2(need)), 0, max_shift)
        unit = row_s * np.exp2(e)                                    # value of code step 1
        q = np.clip(np.rint((g / unit + 3) / 2), 0, 3)
        d = (2 * q - 3) * unit
        err = ((d - g) ** 2).sum(axis=2, keepdims=True)
        if row_clip:  # one clip per row (what a one-scale-per-row kernel format can hold)
            err = err.sum(axis=1, keepdims=True)
        if best is None:
            best, best_err = d, err
        else:
            m = err < best_err
            best = np.where(m, d, best); best_err = np.where(m, err, best_err)
    out = best.reshape(n, k)
    return out, float(np.linalg.norm(out - w) / max(np.linalg.norm(w), 1e-12))


def fake_quant(w, group, bits):
    """w: (N, K) float32 -> dequantized (N, K) float32, plus relative Frobenius error."""
    n, k = w.shape
    assert k % group == 0, (k, group)
    g = w.reshape(n, k // group, group).astype(np.float32)
    mn, mx = g.min(axis=2, keepdims=True), g.max(axis=2, keepdims=True)
    levels = (1 << bits) - 1
    best, best_err = None, None
    for a in np.linspace(0.55, 1.0, 10):
        mid = 0.5 * (mn + mx)
        lo, hi = mid + a * (mn - mid), mid + a * (mx - mid)
        scale = np.maximum(hi - lo, 1e-12) / levels
        q = np.clip(np.rint((g - lo) / scale), 0, levels)
        d = q * scale + lo
        err = ((d - g) ** 2).sum(axis=2, keepdims=True)
        if best is None:
            best, best_err = d, err
        else:
            m = err < best_err
            best = np.where(m, d, best)
            best_err = np.where(m, err, best_err)
    out = best.reshape(n, k)
    rel = float(np.linalg.norm(out - w) / max(np.linalg.norm(w), 1e-12))
    return out, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--group", type=int, default=64)
    ap.add_argument("--bits", type=int, default=2)
    ap.add_argument("--pow2", action="store_true", help="kernel-friendly format: symmetric codes, power-of-two group scales")
    ap.add_argument("--row-clip", action="store_true", help="--pow2 with the clip chosen per row, not per group (the W2P kernel format)")
    a = ap.parse_args()
    t0 = time.time()
    shutil.copyfile(a.src, a.dst)
    r = gguf.GGUFReader(a.dst, "r+")
    n_t, n_p, errs = 0, 0, []
    for t in r.tensors:
        if not SITES.match(t.name):
            continue
        if t.tensor_type not in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.BF16):
            sys.exit(f"{t.name}: unsupported type {t.tensor_type.name}")
        data = t.data  # numpy view into the memmap, shape (N, K)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            w = (data.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            w = data.astype(np.float32)
        w = w.reshape(int(t.shape[1]), int(t.shape[0]))
        dq, rel = fake_quant_pow2(w, a.group, row_clip=a.row_clip) if a.pow2 else fake_quant(w, a.group, a.bits)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            u = dq.astype(np.float32).view(np.uint32)
            u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
            data.view(np.uint16)[...] = u.reshape(data.shape)
        else:
            data[...] = dq.astype(data.dtype).reshape(data.shape)
        n_t += 1; n_p += w.size; errs.append(rel)
    r.data.flush()
    print(f"fake-quantized {n_t} tensors ({n_p / 1e6:.1f} M weights) to {a.bits} bits{' (pow2 group scales)' if a.pow2 else ''}, group {a.group}{', clip per row' if a.row_clip else ''}: "
          f"weight rel-error mean {np.mean(errs):.4f}, max {np.max(errs):.4f}  ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
