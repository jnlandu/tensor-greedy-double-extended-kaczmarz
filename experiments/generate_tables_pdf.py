#!/usr/bin/env python3
"""
generate_tables_pdf.py
Generate a single PDF (experiments/tables/tables.pdf) containing all experiment tables.

Run after run_all_experiments.py — update the data constants at the top when results change.
"""

import os, subprocess, math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTDIR     = os.path.join(SCRIPT_DIR, "tables")
TEX_PATH   = os.path.join(OUTDIR, "tables.tex")
PDF_PATH   = os.path.join(OUTDIR, "tables.pdf")

# ══════════════════════════════════════════════════════════════════════════════
# Results from the last run of run_all_experiments.py
# ══════════════════════════════════════════════════════════════════════════════

METHODS = ["TREK", "TREBK", "TREGBK", "TGDBEK"]

# Table 1 — Dense overdetermined systems (5-trial averages)
T1_N = [20, 30, 40, 50, 60, 80]
T1_IT = {
    "TREK":   [542, 800, 800, 800, 800, 800],
    "TREBK":  [ 36,  66,  95, 126, 167, 259],
    "TREGBK": [ 36,  62,  89, 117, 148, 217],
    "TGDBEK": [ 18,  30,  28,  36,  33,  48],
}
T1_CPU = {
    "TREK":   [2.225, 3.244, 2.947, 2.952, 2.788, 2.855],
    "TREBK":  [1.003, 1.632, 2.153, 2.897, 3.691, 6.118],
    "TREGBK": [1.051, 1.678, 2.368, 3.162, 3.916, 6.451],
    "TGDBEK": [0.859, 1.303, 1.285, 1.708, 1.732, 3.100],
}

# Table 2 — Sparse SuiteSparse systems  (IT, CPU, RSE)
T2_MATS = ["nos5", "ash85", "Cities", "WorldCities"]
T2 = {
    "nos5": {
        "TREK":   (300,  1.239, 9.288e-2),
        "TREBK":  (138,  4.054, 9.888e-7),
        "TREGBK": (106,  3.399, 9.559e-7),
        "TGDBEK": ( 44,  2.828, 9.659e-7),
    },
    "ash85": {
        "TREK":   (300,  1.173, 1.386e-1),
        "TREBK":  (171,  2.544, 9.463e-7),
        "TREGBK": (300,  4.401, 1.028e-6),
        "TGDBEK": (128,  2.964, 9.716e-7),
    },
    "Cities": {
        "TREK":   (6000, 21.075, 3.807e-3),
        "TREBK":  (1916, 15.799, 9.815e-7),
        "TREGBK": (5740, 46.848, 9.881e-7),
        "TGDBEK": (1107, 15.355, 9.783e-7),
    },
    "WorldCities": {
        "TREK":   (6000, 22.366, float("nan")),
        "TREBK":  (   3,  0.155, 7.770e-7),
        "TREGBK": ( 228,  5.413, 8.757e-7),
        "TGDBEK": (   1,  0.070, 5.527e-7),
    },
}

# Tables 3–5 — Image deblurring  (IT, CPU, RSE, PSNR, SSIM)
T3 = {   # flower 200×200×3, noise=1e-2
    "TREK":   (1000,  3.580, 7.885e-1, 13.04, 0.2654),
    "TREBK":  (1000,  9.413, 1.428e-2, 30.19, 0.8405),
    "TREGBK": (1000, 16.637, 1.172e-3, 30.32, 0.8432),
    "TGDBEK": ( 480, 14.441, 9.877e-6, 30.32, 0.8432),
}
T4 = {   # original 200×200×3, noise=1e-1
    "TREK":   (3000,  9.920, 7.175e-1, 12.52, 0.1786),
    "TREBK":  (2577, 24.241, 9.036e-6, 12.13, 0.1582),
    "TREGBK": (1714, 24.862, 9.785e-6, 12.13, 0.1582),
    "TGDBEK": ( 541, 17.832, 9.993e-6, 12.13, 0.1582),
}
T5 = {   # Shepp-Logan 128×128×27, noise=1e-3
    "TREK":   (1000,  4.536, 7.161e-1, 17.75, 0.5822),
    "TREBK":  (1000, 60.832, 5.407e-4, 58.02, 0.9976),
    "TREGBK": (1000, 69.298, 2.437e-5, 58.07, 0.9976),
    "TGDBEK": ( 477, 42.851, 9.878e-6, 58.07, 0.9976),
}

# Table 6 — η sweep on TGDBEK  (eta, IT, CPU)
T6 = [
    (0.1,   1, 0.065),
    (0.2,   1, 0.032),
    (0.3,   1, 0.032),
    (0.4,   1, 0.043),
    (0.5,   6, 0.655),
    (0.6,  18, 0.966),
    (0.7,  37, 1.457),
    (0.8,  57, 1.760),
    (0.9,  88, 2.353),
    (1.0, 362, 7.962),
]

# ══════════════════════════════════════════════════════════════════════════════
# LaTeX helpers
# ══════════════════════════════════════════════════════════════════════════════

def sci(v, prec=3):
    """Format float as LaTeX scientific notation $a.bbb \times 10^{c}$."""
    if v != v:          # NaN
        return r"\text{---}"
    if v == 0:
        return "$0$"
    exp = int(math.floor(math.log10(abs(v))))
    mant = v / 10**exp
    if prec == 0:
        return f"$10^{{{exp}}}$"
    return f"${mant:.{prec}f}\\times 10^{{{exp}}}$"

def bold(s, do_bold=True):
    return r"\textbf{" + str(s) + "}" if do_bold else str(s)

def best_it(method, data_dict):
    """True if this method has the minimum IT among converged methods."""
    its = {m: data_dict[m][0] for m in METHODS}
    # only compare finite RSEs
    rses = {m: data_dict[m][2] for m in METHODS}
    converged = [m for m in METHODS if rses[m] == rses[m]]  # not NaN
    if not converged:
        return False
    return its[method] == min(its[m] for m in converged)

# ══════════════════════════════════════════════════════════════════════════════
# Table builders
# ══════════════════════════════════════════════════════════════════════════════

def table1_it():
    """Table 1a: Iteration counts for the dense experiment."""
    ncols = len(T1_N)
    col_spec = "l" + "r" * ncols
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Number of iterations (IT) for dense overdetermined tensor systems "
        r"$\mathcal{A}\in\mathbb{R}^{500\times n\times 10}$, averaged over 5 independent trials. "
        r"Bold indicates the proposed TGDBEK method. "
        r"``---'' denotes no convergence within the iteration budget.}",
        r"\label{tab:dense-it}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$n={n}$" for n in T1_N) + r" \\",
        r"\midrule",
    ]
    for m in METHODS:
        is_tg = (m == "TGDBEK")
        cells = []
        for j, n in enumerate(T1_N):
            v = T1_IT[m][j]
            s = "---" if v >= 800 and m == "TREK" and j > 0 else str(v)
            cells.append(bold(s, is_tg))
        lines.append(bold(m, is_tg) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table1_cpu():
    """Table 1b: CPU times for the dense experiment."""
    ncols = len(T1_N)
    col_spec = "l" + "r" * ncols
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{CPU time (seconds) for dense overdetermined tensor systems, "
        r"averaged over 5 independent trials.}",
        r"\label{tab:dense-cpu}",
        r"\begin{tabular}{" + col_spec + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$n={n}$" for n in T1_N) + r" \\",
        r"\midrule",
    ]
    for m in METHODS:
        is_tg = (m == "TGDBEK")
        cells = [bold(f"{T1_CPU[m][j]:.3f}", is_tg) for j in range(len(T1_N))]
        lines.append(bold(m, is_tg) + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table2():
    """Table 2: Sparse SuiteSparse results."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Results on sparse tensor systems from SuiteSparse. "
        r"IT: iterations; CPU: seconds; RSE: relative squared error "
        r"$\|\mathcal{X}^{(k)}-\mathcal{X}^*\|_F^2/\|\mathcal{X}^*\|_F^2$. "
        r"``---'' denotes no convergence. $*$ marks the proposed method.}",
        r"\label{tab:sparse}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Matrix & Method & IT & CPU (s) & RSE \\",
        r"\midrule",
    ]
    for i, mat in enumerate(T2_MATS):
        if i > 0:
            lines.append(r"\midrule")
        for m in METHODS:
            it, cpu, rse = T2[mat][m]
            is_tg = (m == "TGDBEK")
            prefix = r"$*$" if is_tg else ""
            it_str  = bold(str(it),  is_tg)
            cpu_str = bold(f"{cpu:.3f}", is_tg)
            rse_str = bold(sci(rse), is_tg)
            mat_col = mat if m == METHODS[0] else ""
            lines.append(
                f"{mat_col} & {prefix}{bold(m, is_tg)} & {it_str} & {cpu_str} & {rse_str} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def image_table(data, label, caption):
    """Generic image deblurring table (IT, CPU, RSE, PSNR, SSIM)."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{" + caption + "}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Method & IT & CPU (s) & RSE & PSNR (dB) & SSIM \\",
        r"\midrule",
    ]
    # find best (converged) IT
    min_it = min(data[m][0] for m in METHODS if data[m][2] < 1e-4)
    for m in METHODS:
        it, cpu, rse, psnr, ssim_v = data[m]
        is_tg  = (m == "TGDBEK")
        is_best = (it == min_it) and (rse < 1e-4)
        prefix  = r"$*$" if is_tg else ""
        lines.append(
            f"{prefix}{bold(m, is_tg)} & {bold(it, is_tg)} & {bold(f'{cpu:.3f}', is_tg)} & "
            f"{bold(sci(rse), is_tg)} & {bold(f'{psnr:.2f}', is_tg)} & "
            f"{bold(f'{ssim_v:.4f}', is_tg)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table6():
    """Table 6: η sweep."""
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Effect of the greedy threshold $\eta$ on TGDBEK. "
        r"Dense system $\mathcal{A}\in\mathbb{R}^{500\times 20\times 10}$, noise $=10^{-3}$.}",
        r"\label{tab:eta}",
        r"\begin{tabular}{crr}",
        r"\toprule",
        r"$\eta$ & IT & CPU (s) \\",
        r"\midrule",
    ]
    for eta, it, cpu in T6:
        lines.append(f"${eta:.1f}$ & {it} & {cpu:.3f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Assemble the full .tex document
# ══════════════════════════════════════════════════════════════════════════════

def build_tex():
    parts = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[margin=2.5cm]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{amsmath,amssymb}",
        r"\usepackage{caption}",
        r"\usepackage{microtype}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\title{\textbf{Experiment Tables}\\",
        r"\large Tensor Greedy Double Block Extended Kaczmarz (TGDBEK)\\",
        r"\normalsize Proposed method marked $*$ / bold}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\thispagestyle{empty}",
        "",
        r"\section*{Example~4.1 — Dense overdetermined tensor systems}",
        r"$\mathcal{A}\in\mathbb{R}^{500\times n\times 10}$, "
        r"$\mathcal{B}\in\mathbb{R}^{500\times 10\times 10}$, "
        r"noise $=10^{-3}$, max iter $=800$, $\eta=0.6$, $\tau=10$, 5 trials.",
        "",
        table1_it(),
        "",
        table1_cpu(),
        "",
        r"\section*{Example~4.2 — Sparse tensor systems (SuiteSparse)}",
        r"Tolerance $10^{-6}$, $\tau=10$, $\eta$ per matrix.",
        "",
        table2(),
        "",
        r"\section*{Examples~4.3--4.5 — Image deblurring}",
        r"Gaussian blur ($\sigma=4$), noise added, tolerance $10^{-5}$, $\eta=0.5$, $\tau=10$.",
        "",
        image_table(
            T3, "tab:flower",
            r"Color image deblurring (flower, $200\times 200\times 3$, noise $=10^{-2}$, "
            r"max iter $=1000$)."),
        "",
        image_table(
            T4, "tab:original",
            r"Color image deblurring (portrait, $200\times 200\times 3$, noise $=10^{-1}$, "
            r"max iter $=3000$)."),
        "",
        image_table(
            T5, "tab:shepp",
            r"Gray MRI image deblurring (Shepp-Logan, $128\times 128\times 27$, noise $=10^{-3}$, "
            r"max iter $=1000$)."),
        "",
        r"\section*{Example~4.6 — Effect of $\eta$ on TGDBEK}",
        "",
        table6(),
        "",
        r"\end{document}",
    ]
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs(OUTDIR, exist_ok=True)

    tex = build_tex()
    with open(TEX_PATH, "w") as f:
        f.write(tex)
    print(f"  Wrote {TEX_PATH}")

    # Compile twice so cross-refs (labels) resolve
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", OUTDIR,
             TEX_PATH],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print("pdflatex stderr:\n", result.stderr[-2000:])
            print("pdflatex stdout:\n", result.stdout[-2000:])
            raise RuntimeError("pdflatex failed")

    # Clean up auxiliary files
    for ext in [".aux", ".log", ".out"]:
        aux = TEX_PATH.replace(".tex", ext)
        if os.path.exists(aux):
            os.remove(aux)

    print(f"  PDF ready: {PDF_PATH}")
