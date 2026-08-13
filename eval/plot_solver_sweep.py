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
"""Render the E7 solver-step sweep figure from the collected sweep results.

Two panels sharing the T axis -- success on the left, latency on the right.
Deliberately NOT a dual-axis chart: two measures on two scales sharing one plot
would invent a correlation that is not in the data.

Reads outputs/solver_sweep/ via collect_solver_sweep.py's own parser, so the
figure and the table cannot drift apart.

    python eval/plot_solver_sweep.py [-o <root>] [-f <out.pdf>]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent

# House style, matching paper/replan_sweep.py and roofline.py so all three
# figures read as one set: DejaVu Serif (ships with matplotlib, so it renders
# identically everywhere), inward ticks, four spines, dotted grid.
#
# Colours are the paper's Okabe-Ito pair, reusing the same two hues in the same
# roles as replan_sweep: blue = the primary/success series, orange = the second.
# Re-validated for this figure against its real surface (white paper):
#   validate_palette.js "#0072B2,#E69F00" --mode light --surface "#ffffff"
#   -> CVD dE 29.2, normal-vision dE 36.2 (both clear), but orange is 2.25:1
#      against white, a contrast WARN. The skill's relief rule applies and is
#      discharged two ways: marker shape carries identity alongside hue
#      (circle vs square), and the caption prints the per-T values, so nothing
#      depends on seeing the orange against the page.
VLA_CPP = "#0072B2"   # blue   (replan_sweep SUCCESS_COLOR)
PYTORCH = "#E69F00"   # orange (replan_sweep RATE_COLOR)
SURFACE = "#ffffff"
GRID = "#bbbbbb"
AXIS = "#333333"

N_ACT = 16  # chunk replay; client/step * N_ACT = per call


def set_research_style() -> None:
    """Verbatim from paper/replan_sweep.py, so the figures share one look."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman No9 L"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 10.5,
        "axes.titlesize": 11,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.linewidth": 0.9,
        "axes.edgecolor": AXIS,
        "axes.axisbelow": True,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.dpi": 220,
        # Keep text as text (TrueType) in PDF/PS so it stays editable.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_collector():
    spec = importlib.util.spec_from_file_location(
        "collect_solver_sweep", REPO_ROOT / "eval" / "collect_solver_sweep.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    # House style keeps all four spines (replan_sweep / roofline).
    ax.grid(True, which="major", ls=":", lw=0.5, color=GRID, alpha=0.7)
    ax.set_axisbelow(True)


def series(ax, x, y, color, marker, ls, label=None, lo=None, hi=None,
           dodge=1.0, line=True):
    """One series. `line=False` gives a dot plot.

    Panel A is trendless repeated measurement, so it gets no connecting line:
    joining those points would draw a zigzag the data does not contain. Marker
    shape doubles the hue, so identity survives the orange's low contrast.
    """
    xd = [v * dodge for v in x]
    if lo is not None:
        err = [[y[i] - lo[i] for i in range(len(y))],
               [hi[i] - y[i] for i in range(len(y))]]
        ax.errorbar(xd, y, yerr=err, fmt="none", ecolor=color,
                    elinewidth=1.3, capsize=3, capthick=1.3, zorder=5)
    if line:
        ax.plot(xd, y, ls=ls, color=color, lw=1.8, zorder=4)
    ax.plot(xd, y, marker, color=color, ms=5, markeredgecolor="white",
            markeredgewidth=0.8, ls="none", zorder=6, label=label)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--root", type=Path,
                    default=REPO_ROOT / "outputs" / "solver_sweep")
    ap.add_argument("-f", "--out", type=Path,
                    default=REPO_ROOT / "docs" / "corl-paper" / "solver_sweep.pdf")
    args = ap.parse_args()

    C = load_collector()
    t_dirs = sorted(args.root.glob("T*"), key=lambda p: int(p.name[1:]))
    t_dirs = [p for p in t_dirs if p.is_dir() and p.name[1:].isdigit()]
    if not t_dirs:
        print(f"ERROR: no T<n> dirs under {args.root}", file=sys.stderr)
        return 1

    T, cpp_sr, cpp_lo, cpp_hi, pt_sr, pt_lo, pt_hi, cpp_ms, pt_ms = ([] for _ in range(9))
    for d in t_dirs:
        cpp, pt = C.read_cell(d / "vla_cpp"), C.read_cell(d / "pytorch")
        if not cpp or not pt:
            print(f"WARNING: skipping incomplete cell {d.name}", file=sys.stderr)
            continue
        T.append(int(d.name[1:]))
        for cell, sr, lo, hi, ms in ((cpp, cpp_sr, cpp_lo, cpp_hi, cpp_ms),
                                     (pt, pt_sr, pt_lo, pt_hi, pt_ms)):
            p = 100.0 * cell["succ"] / cell["total"]
            l, h = C.wilson(cell["succ"], cell["total"])
            sr.append(p); lo.append(l); hi.append(h)
            ms.append(cell["step_ms"] * N_ACT)
    if not T:
        print("ERROR: no complete cells", file=sys.stderr)
        return 1

    set_research_style()
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # -- Panel A: success ---------------------------------------------------
    # Small x dodge so the two Wilson intervals never sit on top of each other.
    series(axA, T, cpp_sr, VLA_CPP, "o", "-", "vla.cpp",
           cpp_lo, cpp_hi, dodge=0.965, line=False)
    series(axA, T, pt_sr, PYTORCH, "s", "--", "PyTorch reference",
           pt_lo, pt_hi, dodge=1.035, line=False)
    # Plain "%" -- matplotlib is not LaTeX here, so "\%" would render literally.
    axA.set_ylabel("LIBERO-Object success (%)")
    axA.set_ylim(84, 101)
    axA.set_yticks([85, 90, 95, 100])

    # -- Panel B: latency ---------------------------------------------------
    series(axB, T, cpp_ms, VLA_CPP, "o", "-")
    series(axB, T, pt_ms, PYTORCH, "s", "--")
    axB.set_ylabel("Latency per prediction (ms)")
    axB.set_ylim(0, 215)
    # Selective direct labels: the endpoints only, where the gap is widest.
    # These are also the contrast relief for the orange series.
    for val, color in ((pt_ms[-1], PYTORCH), (cpp_ms[-1], VLA_CPP)):
        axB.annotate(f"{val:.0f}", (T[-1], val), textcoords="offset points",
                     xytext=(7, -3), ha="left", fontsize=9, color=color)

    for ax, title in ((axA, "Success is flat in $T$; every interval overlaps"),
                      (axB, "Latency grows with $T$; vla.cpp leads throughout")):
        style_axes(ax)
        ax.set_xscale("log", base=2)
        ax.set_xticks(T)
        ax.set_xticklabels([str(t) for t in T])
        ax.minorticks_off()
        ax.set_xlim(T[0] * 0.85, T[-1] * (1.45 if ax is axB else 1.12))
        ax.set_xlabel("Solver steps $T$")
        ax.set_title(title, fontsize=10, pad=6)

    # One frameless legend under both panels, as in replan_sweep.
    h, l = axA.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02), handletextpad=0.5)

    fig.tight_layout(rect=(0, 0.07, 1, 1.0), w_pad=2.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    png = args.out.with_suffix(".png")
    fig.savefig(png, dpi=220, bbox_inches="tight")
    print(f"Wrote {args.out}\nWrote {png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
