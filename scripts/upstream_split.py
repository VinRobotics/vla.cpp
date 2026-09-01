#!/usr/bin/env python3
# Copyright 2026 VinRobotics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Split scripts/patch_ggml_openvino.py into one llama.cpp branch per upstream PR.

Every hunk in the patch script is a generic ggml-openvino fix, so it belongs
upstream rather than in a configure-time rewrite here. This regroups the hunks
into per-PR commits on a clone of llama.cpp master, which is what
docs/UPSTREAMING.md tracks. It only writes to that clone.

    git clone --depth 1 https://github.com/ggml-org/llama.cpp ~/llama.cpp-upstream
    python3 scripts/upstream_split.py [<clone-dir>]

Every hunk must land in exactly one PR; the tail of the output says so.
"""
import importlib.util, os, pathlib, subprocess, sys

REPO = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/llama.cpp-upstream"))
HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("p", HERE / "patch_ggml_openvino.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
E = m.EDITS
D = "ggml/src/ggml-openvino/"


def H(rel, key):
    """Address a hunk by a unique substring of its anchor, not by position.

    These used to be plain indices, and adding a hunk to the front of a file's
    list silently handed every later hunk to the wrong branch -- the tail
    coverage count still read 29/29, because each index was still used once.
    A key that stops matching fails here instead of committing the wrong diff.
    """
    hits = [i for i, (anchor, _) in enumerate(E[rel]) if key in anchor]
    if len(hits) != 1:
        sys.exit(f"upstream_split: key {key!r} matched {len(hits)} hunks in {rel}, expected 1")
    return rel, hits[0]


# branch -> (subject, body, [H(file, anchor-substring), ...])
PRS = [
 ("openvino-naive-cache",
  "openvino: cache the compiled model on the naive path",
  "The dynamic and static paths keep a graph_key-indexed cache of the decoder and\n"
  "the compiled infer request. The naive path has none, so every\n"
  "ggml_backend_graph_compute rebuilds the decoder, re-converts the model and\n"
  "calls compile_model() again. On a graph that is not a decoder-only LLM that is\n"
  "the dominant cost: a 512px SmolVLA vision tower plus action expert goes from\n"
  "22.7 s to 1.4 s per prediction on the CPU plugin.\n\n"
  "A hit rebinds the cached decoder through the existing update_io(), the same way\n"
  "the dynamic path handles freshly built tensors.\n\n"
  "The cache is keyed on naive_key rather than graph_key. graph_key is n_nodes plus\n"
  "the first and last node name, which two graphs of the same size can share, and a\n"
  "compiled model is bound to the shapes it was built for, so a collision returns\n"
  "another graph's answer with no error. naive_key mixes in every node's op, type\n"
  "and shape. The map is bounded and flushed when full.",
  [H(D+"utils.h","struct decoder_runtime_ctx"),H(D+"utils.h","graph_key_hash> decoder_cache"),
   H(D+"utils.h","decoder_cache.clear()"),H(D+"utils.h","enum ggml_status naive_compute"),
   H(D+"utils.cpp","if (!model_is_splitted)"),H(D+"utils.cpp","if (is_naive(cgraph))"),
   H(D+"utils.cpp","enum ggml_status naive_compute")]),

 ("openvino-naive-graph-size-env",
  "openvino: make the naive-path graph-size threshold settable",
  "Graphs under 20 nodes bypass the LLM decoder and translate literally, with\n"
  "static shapes and no KV-cache inference. That literal path is the one that fits\n"
  "a graph which is not a decoder-only transformer, but such a graph is routinely\n"
  "far larger than 20 nodes: a ViT tower is around 450.\n\n"
  "Expose the constant as GGML_OPENVINO_NAIVE_GRAPH_SIZE. Parsed with strtol and\n"
  "rejected loudly if it is not a whole positive number, because atoi turns junk\n"
  "into 0 and that would send every graph down the LLM builder with nothing said.",
  [H(D+"utils.cpp","bool is_naive(ggml_cgraph")]),

 ("openvino-gelu-modes",
  "openvino: map GELU to tanh and add GELU_ERF",
  "ggml has two GELUs: GGML_UNARY_OP_GELU is the tanh approximation and\n"
  "GGML_UNARY_OP_GELU_ERF is the exact one. The table maps GELU onto ov's Gelu,\n"
  "which defaults to erf, and has no entry for GELU_ERF at all. So the tanh op is\n"
  "computed as erf, and a graph using the erf op cannot run.\n\n"
  "Small per node, but a vision tower has dozens and it compounds: on a ggml graph\n"
  "with a ViT encoder, fixing the mode moved two models from visibly wrong output\n"
  "to within 1.3e-3 of the CPU-backend reference.",
  [H(D+"openvino/op_table.cpp","namespace ov {"),H(D+"openvino/op_table.cpp","{\"GGML_UNARY_OP_GELU\",")]),

 ("openvino-multiple-inp-pos",
  "openvino: stop distinct position inputs aliasing each other",
  "Every tensor feeding a ROPE's second input is renamed to one graph parameter\n"
  "called inp_pos, because a llama.cpp graph has exactly one. add_rope_sin_cos()\n"
  "then builds a single shared sin/cos table from it.\n\n"
  "A graph with several position tensors -- a prefill, a full and a rebased one --\n"
  "has them alias each other, and every ROPE takes the table built from whichever\n"
  "won. Shape inference then fails:\n\n"
  "  Multiply (Split[1]:f32[1,113,5,32], Multiply[0]:f32[1,50,1,32])\n"
  "  Argument shapes are inconsistent.\n\n"
  "When the graph has more than one, keep each tensor's own name. Nothing is then\n"
  "called inp_pos, so translate_rope() builds sin/cos per op from its own position\n"
  "input. Single-position graphs are untouched.\n\n"
  "Guard the free get_tensor_graph_input_ov_name() as well as the GgmlOvDecoder\n"
  "member: the free function is the one compute_model_inputs() and\n"
  "set_input_output() actually call, and the member currently has no callers.",
  [H(D+"ggml-decoder.h","get_graph_input_ov_name"),H(D+"ggml-decoder.h","m_cgraph = nullptr"),
   H(D+"ggml-decoder.cpp","is_inp_pos(tensor, op)"),H(D+"ggml-decoder.cpp","compute_op_case(const ggml_tensor")]),

 ("openvino-reshape-op-case",
  "openvino: narrow the RESHAPE op_case 3 guard",
  "Case 3 is the KV-cache flatten, [512,1024,1,1] -> [1,524288,1,1], and it emits a\n"
  "shape with -1 in dim 2 and 1 in dim 3. Its guard only tests\n"
  "src->ne[0]*ne[1]*ne[2] == node->ne[1], which also matches the kernel reshape\n"
  "inside ggml_conv_2d ([16,16,3,768] -> [768,768]) and rewrites it to the wrong\n"
  "shape. The real case always has node->ne[0] == 1; requiring that sends the conv\n"
  "kernel to case 6, the plain reshape.",
  [H(D+"ggml-decoder.cpp","== node->ne[1]")]),

 ("openvino-gemm-double-eltwise",
  "openvino: do not stack two elementwise adds on a GEMM",
  "The GPU plugin folds elementwise ops into the preceding GEMM as post-ops. Given\n"
  "ADD(ADD(residual, GEMM), graph_input) it folds both and the second operand is\n"
  "silently lost: the result equals the inner add, as though the outer one never\n"
  "ran. Nothing is logged.\n\n"
  "A decoder-only LLM graph never builds that chain - one residual add per\n"
  "sub-block - but a graph that adds a vision tower's features on top of an FFN\n"
  "residual does, and three such models produced badly wrong output on GPU while\n"
  "matching the CPU plugin to 1e-4.\n\n"
  "Addition is associative, so re-hang the outer add on the inner one's non-GEMM\n"
  "operand; the GEMM keeps exactly one post-op. op_case 2/3 records which operand\n"
  "of the inner add is the GEMM, since the order is not fixed.\n\n"
  "Same fusion path as the broadcast-DIV defect already handled in supports_op.",
  [H(D+"ggml-decoder.cpp","case GGML_OP_ADD: {"),
   H(D+"openvino/op/add.cpp","auto input_0 = process_view_input_new(context, 0);")]),

 ("openvino-permute-op-case",
  "openvino: require a ROPE before taking PERMUTE op_case 2",
  "op_case 2 rewrites the tensor as [n_seq, -1, n_heads, head_size] and only then\n"
  "transposes, which is correct for llama.cpp's rope'd query and nothing else. The\n"
  "classifier reaches it for ANY permute whose source is a view of a non-leaf.\n\n"
  "A DiT cross-attention V built as ggml_permute(view, 1,2,0,3) over a fused KV\n"
  "projection is therefore reshaped into a shape unrelated to it and comes out with\n"
  "its elements rearranged, while a sibling K using permute(0,2,1,3) survives the\n"
  "same rewrite -- 139% wrong against 0.04%, with no error reported.\n\n"
  "Walk the view/reshape/cont chain and require a ROPE at the end; every other\n"
  "permute falls to op_case 1, the plain transpose.",
  [H(D+"ggml-decoder.cpp","rope'ed query tensor")]),

 ("openvino-sdpa-kv-f16",
  "openvino: convert K/V to F16 alongside Q in flash_attn_ext",
  "The translator converts Q, the mask and the scale to F16 because llama.cpp's KV\n"
  "cache already is. A caller that keeps K/V in F32 hits OpenVINO's SDPA rejecting\n"
  "mixed input types (\"Mixed input types are not supported\"). Converting K/V too\n"
  "matches the precision the translator has already chosen for the other operands.",
  [H(D+"openvino/op/flash_attn_ext.cpp","q_f32, ov::element::f16")]),

 ("openvino-view-input-rank",
  "openvino: give a folded weight its full rank before slicing",
  "A ggml tensor that is 2-D folds in as a rank-2 ov constant, which is what a GEMM\n"
  "operand wants, but process_view_input_new() indexes a viewed tensor at its full\n"
  "ggml rank, so the slice axis lands outside it:\n\n"
  "  Slice (Constant blk.0.attn_in.weight[0]:bf16[2688,896], ...)\n"
  "  Axis 2 out of the tensor rank range [-2, 1].\n\n"
  "Reached by viewing Q, K and V out of one fused attn_in weight. Left-pad the\n"
  "input with leading 1s, which is the shape ggml gave it anyway.",
  [H(D+"openvino/utils.cpp","openvino/op/transpose.hpp"),H(D+"openvino/utils.cpp","get_view_input_size(input_index)")]),

 ("openvino-concat-rank",
  "openvino: align CONCAT input ranks",
  "The same rank-2 folded constants reach CONCAT, which unlike the broadcasting\n"
  "elementwise ops needs both inputs at the graph's rank or the axis falls outside\n"
  "them. Hit by concatenating a CLS weight onto 4-D patch embeddings, and by\n"
  "concatenating a precomputed time tile onto a 4-D activation. Left-pad the\n"
  "shorter input with leading 1s before picking the axis.",
  [H(D+"openvino/op/concat.cpp","openvino/op/concat.hpp"),H(D+"openvino/op/concat.cpp","#include <memory>"),
   H(D+"openvino/op/concat.cpp","rank - 1 - ggml_dim")]),

 ("openvino-imrope-sections",
  "openvino: bound the interleaved-mrope sector cycle by sections",
  "ggml's IMROPE cycles t/h/w by sector % 3, but only while the sector is inside\n"
  "3 * sections[k]; past that it falls through to the fourth position stream\n"
  "(ggml_rope_cache_init in ggml/src/ggml-cpu/ops.cpp). The translator cycled\n"
  "unconditionally, so with sections {24,20,20,0} and n_dims 128, sectors 61 and 62\n"
  "took h and w instead of e.\n\n"
  "No measurable output change on the graphs tested, because the fourth stream\n"
  "happened to carry the same positions as the first. Submitted because it removes\n"
  "a divergence from the ggml reference, not because a measurement demanded it.",
  [H(D+"openvino/utils.cpp","#include <memory>"),H(D+"openvino/utils.cpp","gather_indices(n_dims_half)")]),

 ("openvino-gpu-precision-env",
  "openvino: expose the GPU plugin's inference precision",
  "The GPU plugin computes in f16 unless told otherwise, which is most of why it is\n"
  "fast, and for nearly every graph that is the right trade.\n\n"
  "It is not the right trade for a graph carrying a long serial chain, such as a\n"
  "flow-matching denoise loop unrolled inside a single graph: the error compounds\n"
  "across every step with nothing to reset it, and a saturating output channel can\n"
  "then cross its threshold in the wrong place. One such model lands 4e-2 from an\n"
  "F32 reference on GPU and 6.5e-5 with this set to f32.\n\n"
  "Exposed rather than forced -- f32 costs roughly 3x -- and defaulted to f16, so\n"
  "existing behaviour is unchanged unless the variable is set.",
  [H(D+"ggml-openvino-extra.cpp","GGML_OPENVINO_LOG_UNSUPPORTED_OPS"),
   H(D+"ggml-openvino-extra.cpp","} else if (cache_dir && strlen(cache_dir) > 0) {")]),

 ("openvino-intel-opencl-platform",
  "openvino: select the Intel OpenCL platform for the GPU remote context",
  "GGML_OPENVINO_DEVICE=GPU builds an OpenVINO remote context on an OpenCL queue and\n"
  "takes the first platform the ICD loader reports. With more than one runtime\n"
  "installed (an NVIDIA card beside the Intel iGPU, POCL, Rusticl) that is whichever\n"
  "/etc/OpenCL/vendors/*.icd sorted first, and the GPU plugin only accepts an Intel\n"
  "context. It aborts at startup with \"Incompatible OpenCL runtime: program is not\n"
  "in expected ELF format\".\n\n"
  "Select by CL_PLATFORM_VENDOR instead. Single-runtime hosts are unaffected.",
  [H(D+"ggml-openvino-extra.cpp","#include <optional>"),H(D+"ggml-openvino-extra.cpp","ggml_openvino_device_config::init"),
   H(D+"ggml-openvino-extra.cpp","cl_int err;"),H(D+"ggml-openvino-extra.cpp","clEnqueueMemFillINTEL_fn"),
   H(D+"ggml-openvino-extra.cpp","clEnqueueMemcpyINTEL_fn")]),
]

def git(*a):
    r = subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True)
    if r.returncode: sys.exit(f"git {' '.join(a)} failed:\n{r.stderr}")
    return r.stdout

base = git("rev-parse", "HEAD").strip()
used = set()
for branch, subject, body, hunks in PRS:
    git("checkout", "-q", "-B", branch, base)
    for rel, idx in hunks:
        used.add((rel, idx))
        anchor, repl = E[rel][idx]
        # The marker is vla.cpp provenance; upstream comments should not carry it.
        repl = repl.replace('vla.cpp: ', '')
        f = REPO / rel
        t = f.read_text()
        if repl in t: continue
        if t.count(anchor) != 1:
            sys.exit(f"{branch}: anchor {idx} in {rel} matched {t.count(anchor)} times")
        f.write_text(t.replace(anchor, repl, 1))
    git("add", "-A")
    git("commit", "-q", "-m", subject, "-m", body)
    print(f"{branch:38s} {git('rev-parse','--short','HEAD').strip()}  {len(hunks)} hunk(s)")

git("checkout", "-q", base)
total = sum(len(v) for v in E.values())
missing = [(r, i) for r, v in E.items() for i in range(len(v)) if (r, i) not in used]
print(f"\n{len(used)}/{total} hunks assigned to {len(PRS)} branches")
if missing: print("UNASSIGNED:", missing)
