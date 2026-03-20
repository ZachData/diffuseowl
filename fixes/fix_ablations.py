#!/usr/bin/env python3
"""
fixes/fix_ablations.py

Runs every ablation that works against the existing activation cache — no new
VLLM generation required. Each ablation trains a fresh probe and compares its
AUROC-vs-N curve against the paper's Table 1 (0.621 → 0.966).

ABLATIONS:
  A1  paper_replicate      Paper's exact methodology on existing cache:
                           shrinking test set, n_trials=1, paper alpha grid.
                           Should reproduce reported values if cache is intact.

  A2  fixed_eval           Fix 2 only: constant holdout (20%), n_trials=5.
                           Same activations as A1, better evaluation protocol.

  A3  wide_alpha           Fix 2 + wider C grid covering both regularization
                           extremes. Reveals if paper's skewed alpha range
                           suppressed or inflated AUROC.

  A4  corrected_probe      All eval fixes (A3) applied to yes_no position —
                           trains on what the code *actually* probed.
                           This is the paper's real result with honest eval.

  A5  letter_position      All eval fixes applied to the corrected letter
                           position. Requires act_letter activations in cache.
                           This is what the paper *claimed* to measure.

  A6  benign_baseline      Trains probe on benign-prompt activations (no
                           sandbagging). If AUROC ≈ A4/A5, the signal is
                           answer-quality, not deception-specific.
                           Requires benign activation cache.

  A7  position_sweep       Trains probe at all available positions for layer 30.
                           Shows full picture: letter, yes_no, last, letter±1.

WHAT "CORRECTED PROBE" MEANS:
  The paper trained a probe it believed was on the letter position. Due to
  Bugs 1+2, it was actually on the yes/no token. A corrected probe means:
    - A4: same yes/no activations, but with fixed evaluation (constant holdout,
          more trials, wider alpha) — the honest version of what was measured
    - A5: letter-position activations with fixed evaluation — what was claimed

USAGE:
    python fixes/fix_ablations.py \\
        --cache-dir experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered \\
        --layer 30

    # Include benign baseline (needs benign cache)
    python fixes/fix_ablations.py --cache-dir ... \\
        --benign-cache-dir experiments/gemma-3-12b/cached_activations/benign/unified_n5000_unfiltered

    # Fast mode: fewer N values and trials
    python fixes/fix_ablations.py --cache-dir ... --fast

    # Run specific ablations only
    python fixes/fix_ablations.py --cache-dir ... --ablations A1 A4 A5

OUTPUTS (fixes/output/):
    ablations_report.txt     Master comparison table + per-ablation detail
    ablations.json           Machine-readable; same N/AUROC structure as
                             original auroc_vs_n_*_summary.json files
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

OUTPUT_DIR = Path("fixes/output")

# Paper's Table 1 — the baseline every ablation is compared against
PAPER_AUROC = {
    16: 0.621, 32: 0.764, 64: 0.828, 128: 0.888,
    256: 0.909, 512: 0.936, 1024: 0.948, 2048: 0.956, 4500: 0.966,
}

# N values to test — matches paper Table 1
N_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4500]

# Appendix B: paper's alpha range (as Ridge alpha; C = 1/alpha for LR)
PAPER_C_VALUES = [1/a for a in [1e2, 1e4, 1e5, 1e6, 1e7]]   # [0.01, 1e-4, 1e-5, 1e-6, 1e-7]

# Wider grid covering both ends
WIDE_C_VALUES = [1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4]


# ─────────────────────────────────────────────────────────────────────────────
# Core probe runner
# ─────────────────────────────────────────────────────────────────────────────

def run_probe_curve(
    activations: np.ndarray,
    labels: np.ndarray,
    n_values: list,
    n_trials: int,
    c_values: list,
    fixed_holdout: bool,
    test_fraction: float = 0.20,
    random_state: int = 42,
    label: str = "",
) -> dict:
    """
    Train logistic regression probes across N values.

    Args:
        activations:    (n_samples, hidden_dim)
        labels:         (n_samples,) binary
        n_values:       training sizes to sweep
        n_trials:       trials per N
        c_values:       C grid for logistic regression (cross-validated)
        fixed_holdout:  True = constant 20% holdout (Fix 2)
                        False = shrinking test set (paper's bug)
        test_fraction:  fraction held out when fixed_holdout=True
        random_state:   base random seed
        label:          name for progress output

    Returns:
        dict with auroc_by_n, metadata, best_c_by_n
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score

    total_n = len(labels)

    if fixed_holdout:
        np.random.seed(random_state)
        pool       = np.random.permutation(total_n)
        test_size  = int(total_n * test_fraction)
        test_idx   = pool[:test_size]
        train_pool = pool[test_size:]
    else:
        test_size  = None
        train_pool = None

    auroc_by_n  = {}
    best_c_by_n = {}

    feasible = [n for n in n_values
                if (fixed_holdout and n <= len(train_pool))
                or (not fixed_holdout and n < total_n)]

    for n in feasible:
        trial_aurocs = []
        trial_cs     = []

        for trial in range(n_trials):
            np.random.seed(random_state + trial)

            if fixed_holdout:
                chosen  = np.random.choice(train_pool, size=n, replace=False)
                X_train = activations[chosen]
                y_train = labels[chosen]
                X_test  = activations[test_idx]
                y_test  = labels[test_idx]
            else:
                idx     = np.random.permutation(total_n)
                X_train = activations[idx[:n]]
                y_train = labels[idx[:n]]
                X_test  = activations[idx[n:]]
                y_test  = labels[idx[n:]]

            if len(np.unique(y_train)) < 2 or len(X_test) == 0:
                continue
            if len(np.unique(y_test)) < 2:
                continue

            clf = LogisticRegressionCV(
                Cs=c_values,
                cv=min(5, int(n * 0.8) // max(1, len(np.unique(y_train)))),
                max_iter=1000,
                solver="lbfgs",
                scoring="roc_auc",
                random_state=random_state,
                n_jobs=-1,
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            trial_aurocs.append(roc_auc_score(y_test, proba))
            trial_cs.append(float(clf.C_[0]))

        if trial_aurocs:
            auroc_by_n[n] = {
                "mean":   float(np.mean(trial_aurocs)),
                "std":    float(np.std(trial_aurocs)),
                "trials": [float(a) for a in trial_aurocs],
            }
            best_c_by_n[n] = {
                "mean_c": float(np.mean(trial_cs)),
                "trials": trial_cs,
            }

        sys.stdout.write(
            f"\r  {label}  N={n:>5}  AUROC={auroc_by_n[n]['mean']:.3f}±{auroc_by_n[n]['std']:.3f}  "
            f"best_C={best_c_by_n[n]['mean_c']:.1e}      "
        )
        sys.stdout.flush()

    print()

    return {
        "auroc_by_n":    auroc_by_n,
        "best_c_by_n":   best_c_by_n,
        "fixed_holdout": fixed_holdout,
        "test_size":     test_size if fixed_holdout else None,
        "n_trials":      n_trials,
        "c_values":      c_values,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cache loading
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Cache loading
# ─────────────────────────────────────────────────────────────────────────────

def _activation_candidates(cache_dir: Path, layer: int, position: str) -> list:
    """Return all plausible paths for an activation file, most likely first."""
    candidates = []
    for fmt in (f"{layer:02d}", str(layer)):
        candidates.append(cache_dir / f"activations_layer{fmt}_pos-{position}.npy")
    return candidates


def load_activations(cache_dir: Path, layer: int, position: str) -> Optional[np.ndarray]:
    for p in _activation_candidates(cache_dir, layer, position):
        if p.exists():
            return np.load(p)
    return None


def load_labels(cache_dir: Path) -> Optional[np.ndarray]:
    for name in ("labels.json", "labels.npy"):
        p = cache_dir / name
        if p.exists():
            if name.endswith(".json"):
                with open(p) as f:
                    return np.array(json.load(f))
            else:
                return np.load(p)
    return None


def available_positions(cache_dir: Path, layer: int) -> list:
    """Find all positions for which activation files exist at this layer."""
    found = set()
    for fmt in (f"{layer:02d}", str(layer)):
        prefix = f"activations_layer{fmt}_pos-"
        for p in cache_dir.glob(f"{prefix}*.npy"):
            pos = p.name[len(prefix):-4]
            found.add(pos)
    return sorted(found)


def available_layers(cache_dir: Path) -> list:
    """Find all layers for which any activation file exists."""
    layers = set()
    for p in cache_dir.glob("activations_layer*_pos-*.npy"):
        part = p.name.split("layer")[1].split("_")[0]
        try:
            layers.add(int(part))
        except ValueError:
            pass
    return sorted(layers)


def diagnose_cache(cache_dir: Path, layer: int) -> str:
    """
    Print a full inventory of what's in the cache directory.
    Called when activations are not found so the user knows exactly what exists.
    """
    lines = [f"  cache_dir: {cache_dir.resolve()}"]
    if not cache_dir.exists():
        lines.append("  ERROR: directory does not exist")
        return "\n".join(lines)

    all_files = sorted(cache_dir.iterdir())
    npy_files  = [f for f in all_files if f.suffix == ".npy"]
    json_files = [f for f in all_files if f.suffix == ".json"]
    act_files  = [f for f in npy_files if f.name.startswith("activations_")]

    lines.append(f"  total files: {len(all_files)}")
    lines.append(f"  .npy files:  {len(npy_files)}")
    lines.append(f"  .json files: {len(json_files)}")
    lines.append(f"  activation .npy files: {len(act_files)}")

    if act_files:
        lines.append("  activation files found:")
        for f in act_files:
            lines.append(f"    {f.name}")
    else:
        lines.append("  NO activation .npy files found.")
        lines.append("  The activation cache was not generated or is in a different directory.")
        lines.append("  To generate activations, run from project root:")
        lines.append("    python scripts/cache_lie_detector_activations.py --filtered")
        layers_found = available_layers(cache_dir)
        if layers_found:
            lines.append(f"  layers with activations: {layers_found}")
        else:
            lines.append("  Other .npy files present:")
            for f in npy_files[:10]:
                lines.append(f"    {f.name}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_comparison_table(ablation_results: dict, n_values: list) -> str:
    """
    Master table: all ablations side by side vs paper.
    Columns: N | paper | A1 | A2 | A3 | A4 | A5 | A6 | delta(best vs paper)
    """
    lines = []

    available = [k for k, v in ablation_results.items() if v is not None]

    # header
    header = f"  {'N':>6}  {'paper':>7}"
    for k in available:
        header += f"  {k:>9}"
    header += f"  {'best_delta':>11}"
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for n in n_values:
        paper = PAPER_AUROC.get(n)
        row   = f"  {n:>6}  {paper:>7.3f}" if paper else f"  {n:>6}  {'N/A':>7}"

        best_auroc = 0.0
        for k in available:
            d = ablation_results[k]["auroc_by_n"].get(n) if ablation_results[k] else None
            if d:
                row += f"  {d['mean']:>7.3f}±{d['std']:.2f}"
                if d["mean"] > best_auroc:
                    best_auroc = d["mean"]
            else:
                row += f"  {'N/A':>9}"

        if paper and best_auroc > 0:
            row += f"  {best_auroc - paper:>+11.3f}"
        else:
            row += f"  {'N/A':>11}"

        lines.append(row)

    return "\n".join(lines)


def format_ablation_detail(key: str, result: dict) -> str:
    if result is None:
        return f"{key}  SKIPPED"

    lines = []
    lines.append(
        f"{key}  fixed_holdout={result['fixed_holdout']}  "
        f"n_trials={result['n_trials']}  "
        f"test_size={result.get('test_size', 'shrinking')}  "
        f"C_range=[{min(result['c_values']):.1e}, {max(result['c_values']):.1e}]"
    )
    lines.append(
        f"  {'N':>6}  {'auroc':>7}  {'std':>6}  {'best_C':>8}  {'vs_paper':>9}"
    )
    lines.append(f"  {'─'*44}")

    for n in N_VALUES:
        d = result["auroc_by_n"].get(n)
        if d is None:
            lines.append(f"  {n:>6}  {'N/A':>7}")
            continue
        c = result["best_c_by_n"].get(n, {})
        paper = PAPER_AUROC.get(n)
        vs = f"{d['mean'] - paper:+.3f}" if paper else "N/A"
        lines.append(
            f"  {n:>6}  {d['mean']:>7.3f}  {d['std']:>6.3f}  "
            f"{c.get('mean_c', 0):>8.1e}  {vs:>9}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

ABLATION_DESCRIPTIONS = {
    "A1": "paper_replicate     shrinking_test_set  n_trials=1  paper_C_grid",
    "A2": "fixed_eval          fixed_holdout_20%   n_trials=5  paper_C_grid",
    "A3": "wide_alpha          fixed_holdout_20%   n_trials=5  wide_C_grid",
    "A4": "corrected_probe     yes_no_position     fixed_holdout  wide_C — what was *actually* measured",
    "A5": "letter_position     letter_position     fixed_holdout  wide_C — what was *claimed* to be measured",
    "A6": "benign_baseline     benign_activations  fixed_holdout  wide_C — no sandbagging",
    "A7": "position_sweep      all_positions       fixed_holdout  wide_C  N=4500_only",
}


def main():
    parser = argparse.ArgumentParser(
        description="Ablation suite for the diffuse lie detector probe experiment"
    )
    parser.add_argument("--cache-dir", required=True,
        help="Path to lie_detector unified activation cache dir")
    parser.add_argument("--benign-cache-dir", default=None,
        help="Path to benign unified activation cache dir (for A6)")
    parser.add_argument("--layer", type=int, default=30)
    parser.add_argument("--ablations", nargs="+",
        choices=["A1", "A2", "A3", "A4", "A5", "A6", "A7"],
        default=["A1", "A2", "A3", "A4", "A5", "A7"],
        help="Which ablations to run (default: all except A6 unless --benign-cache-dir given)")
    parser.add_argument("--fast", action="store_true",
        help="Fewer N values and trials for quick turnaround")
    parser.add_argument("--n-trials", type=int, default=5)
    args = parser.parse_args()

    try:
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import roc_auc_score
    except ImportError:
        print("[!] scikit-learn required: pip install scikit-learn")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    # ── Load labels ───────────────────────────────────────────────────────────
    labels = load_labels(cache_dir)
    if labels is None:
        print(f"[!] labels not found in {cache_dir}")
        print(diagnose_cache(cache_dir, args.layer))
        sys.exit(1)

    total_n = len(labels)

    # ── Build N values from all available data ────────────────────────────────
    # Include paper's Table 1 values plus the maximum trainable N
    # (80% of total = max train pool with 20% holdout)
    max_train = int(total_n * 0.80)
    if args.fast:
        n_values = [n for n in [16, 64, 256, 1024] if n < max_train] + [max_train]
        n_trials = 2
    else:
        n_values = [n for n in N_VALUES if n < max_train] + [max_train]
        n_trials = args.n_trials
    n_values = sorted(set(n_values))

    print(f"[*] Cache: {cache_dir.resolve()}")
    print(f"[*] n={total_n}  correct={int(np.sum(labels))} ({100*np.mean(labels):.1f}%)")
    print(f"[*] Layer: {args.layer}")
    print(f"[*] N values: {n_values}")
    print(f"[*] n_trials: {n_trials}")

    # ── Inventory available activation files ───────────────────────────────────
    positions_available = available_positions(cache_dir, args.layer)
    layers_available    = available_layers(cache_dir)
    print(f"[*] Layers with activations: {layers_available or 'none found'}")
    print(f"[*] Positions at layer {args.layer}: {positions_available or 'none found'}")

    if not positions_available:
        print("\n[!] No activation .npy files found for layer", args.layer)
        print(diagnose_cache(cache_dir, args.layer))
        print("\n  Ablations A1-A5 and A7 require activation files.")
        print("  Generate them from the project root with:")
        print("    python scripts/cache_lie_detector_activations.py --filtered")
        print("  Then re-run this script.")

    if args.benign_cache_dir:
        args.ablations = list(set(args.ablations) | {"A6"})
    print(f"[*] Running ablations: {sorted(args.ablations)}")

    results = {}
    to_run  = sorted(args.ablations)

    def _load_or_skip(ablation_id: str, position: str, cache: Path = cache_dir):
        act = load_activations(cache, args.layer, position)
        if act is None:
            candidates = _activation_candidates(cache, args.layer, position)
            print(f"  [skip] activation file not found. Tried:")
            for c in candidates:
                print(f"    {c}")
        return act

    # ── A1: paper replication ─────────────────────────────────────────────────
    if "A1" in to_run:
        print("\n[A1] Replicating paper methodology (shrinking test, n_trials=1, paper C grid)...")
        act = _load_or_skip("A1", "letter")
        if act is None:
            results["A1"] = None
        else:
            results["A1"] = run_probe_curve(
                act, labels, n_values, n_trials=1,
                c_values=PAPER_C_VALUES,
                fixed_holdout=False, label="A1",
            )

    # ── A2: fixed eval only ───────────────────────────────────────────────────
    if "A2" in to_run:
        print("\n[A2] Fixed holdout + n_trials (same C grid as paper)...")
        act = _load_or_skip("A2", "letter")
        if act is None:
            results["A2"] = None
        else:
            results["A2"] = run_probe_curve(
                act, labels, n_values, n_trials=n_trials,
                c_values=PAPER_C_VALUES,
                fixed_holdout=True, label="A2",
            )

    # ── A3: fixed eval + wide alpha ───────────────────────────────────────────
    if "A3" in to_run:
        print("\n[A3] Fixed holdout + wide C grid...")
        act = _load_or_skip("A3", "letter")
        if act is None:
            results["A3"] = None
        else:
            results["A3"] = run_probe_curve(
                act, labels, n_values, n_trials=n_trials,
                c_values=WIDE_C_VALUES,
                fixed_holdout=True, label="A3",
            )

    # ── A4: corrected probe (yes_no) ──────────────────────────────────────────
    if "A4" in to_run:
        print("\n[A4] Corrected probe — yes_no position (what was *actually* probed)...")
        act = load_activations(cache_dir, args.layer, "yes_no")
        if act is None:
            # fall back to letter (identical due to the bug, but label it)
            act = load_activations(cache_dir, args.layer, "letter")
            if act is not None:
                print("  [info] yes_no activations not cached separately;")
                print("         using letter activations (== yes_no due to Bug 1+2)")
        if act is None:
            print(f"  [skip] neither yes_no nor letter activations found")
            results["A4"] = None
        else:
            results["A4"] = run_probe_curve(
                act, labels, n_values, n_trials=n_trials,
                c_values=WIDE_C_VALUES,
                fixed_holdout=True, label="A4",
            )

    # ── A5: letter position ───────────────────────────────────────────────────
    if "A5" in to_run:
        print("\n[A5] Letter position — what the paper *claimed* to probe...")
        act = _load_or_skip("A5", "letter")
        if act is None:
            results["A5"] = None
        else:
            # Check token quality
            tok_path = cache_dir / "probed_tokens_pos-letter.json"
            token_note = "cache not re-extracted — letter pos == yes_no (sabotaged)"
            if tok_path.exists():
                with open(tok_path) as f:
                    toks = json.load(f)
                letter_pct = 100 * sum(1 for t in toks if t and t.strip() in ["A","B","C","D"]) / max(len(toks), 1)
                if letter_pct > 50:
                    token_note = f"OK: {letter_pct:.1f}% of probed tokens are A/B/C/D"
                else:
                    token_note = f"WARNING: only {letter_pct:.1f}% letter tokens — sabotaged cache, re-extract with Fix 1 first"
            print(f"  token check: {token_note}")
            results["A5"] = run_probe_curve(
                act, labels, n_values, n_trials=n_trials,
                c_values=WIDE_C_VALUES,
                fixed_holdout=True, label="A5",
            )
            results["A5"]["token_note"] = token_note

    # ── A6: benign baseline ───────────────────────────────────────────────────
    if "A6" in to_run:
        print("\n[A6] Benign baseline — probe trained on non-sandbagging activations...")
        if args.benign_cache_dir is None:
            print("  [skip] --benign-cache-dir not provided")
            results["A6"] = None
        else:
            benign_dir = Path(args.benign_cache_dir)
            ben_labels = load_labels(benign_dir)
            ben_act    = load_activations(benign_dir, args.layer, "letter")
            if ben_act is None or ben_labels is None:
                print(f"  [skip] benign activations/labels not found")
                print(diagnose_cache(benign_dir, args.layer))
                results["A6"] = None
            else:
                ben_n      = len(ben_labels)
                ben_max_n  = int(ben_n * 0.80)
                ben_nvals  = sorted(set([n for n in n_values if n < ben_max_n] + [ben_max_n]))
                print(f"  benign n={ben_n}  correct={int(np.sum(ben_labels))}")
                results["A6"] = run_probe_curve(
                    ben_act, ben_labels, ben_nvals, n_trials=n_trials,
                    c_values=WIDE_C_VALUES,
                    fixed_holdout=True, label="A6",
                )

    # ── A7: position sweep ────────────────────────────────────────────────────
    if "A7" in to_run:
        print(f"\n[A7] Position sweep at N={max_train}, layer={args.layer}...")
        if not positions_available:
            print("  [skip] no activation files found")
            results["A7"] = {"position_sweep": {}, "n": max_train}
        else:
            pos_results = {}
            for pos in positions_available:
                act = load_activations(cache_dir, args.layer, pos)
                if act is None:
                    continue
                r = run_probe_curve(
                    act, labels, [max_train], n_trials=n_trials,
                    c_values=WIDE_C_VALUES,
                    fixed_holdout=True, label=f"A7/{pos}",
                )
                d = r["auroc_by_n"].get(max_train)
                if d:
                    pos_results[pos] = {"mean": d["mean"], "std": d["std"]}
            results["A7"] = {"position_sweep": pos_results, "n": max_train}

    # ── Build report ──────────────────────────────────────────────────────────
    sections = []

    sections.append("ablations")
    for k in sorted(args.ablations):
        sections.append(f"  {k}  {ABLATION_DESCRIPTIONS.get(k, '')}")
    sections.append(
        f"  note  n_total={total_n}  max_train={max_train}  "
        f"paper_C_range=[{min(PAPER_C_VALUES):.1e},{max(PAPER_C_VALUES):.1e}]  "
        f"wide_C_range=[{min(WIDE_C_VALUES):.1e},{max(WIDE_C_VALUES):.1e}]"
    )
    sections.append("")

    curve_results = {k: v for k, v in results.items()
                     if k != "A7" and v is not None and "auroc_by_n" in (v or {})}
    if curve_results:
        sections.append("comparison_table  (mean±std vs paper Table 1)")
        sections.append(format_comparison_table(curve_results, n_values))
        sections.append("")

    for k in sorted(k for k in results if k != "A7"):
        sections.append(format_ablation_detail(k, results.get(k)))
        sections.append("")

    if "A7" in results and results["A7"] is not None:
        ps = results["A7"]["position_sweep"]
        n  = results["A7"]["n"]
        if ps:
            sections.append(f"A7_position_sweep  N={n}  layer={args.layer}")
            sections.append(f"  {'position':<14} {'auroc':>7}  {'std':>6}  {'vs_paper_4500':>14}")
            sections.append(f"  {'─'*46}")
            paper_4500 = PAPER_AUROC.get(4500)
            for pos, d in sorted(ps.items(), key=lambda x: -x[1]["mean"]):
                vs = f"{d['mean'] - paper_4500:+.3f}" if paper_4500 else "N/A"
                sections.append(f"  {pos:<14} {d['mean']:>7.3f}  {d['std']:>6.3f}  {vs:>14}")
        else:
            sections.append(f"A7_position_sweep  N={n}  layer={args.layer}")
            sections.append("  no activation files found — run cache_lie_detector_activations.py first")
        sections.append("")

    # Check whether any ablation actually ran
    curve_ran = any(
        v is not None and "auroc_by_n" in (v or {}) and v["auroc_by_n"]
        for k, v in results.items() if k != "A7"
    )
    sweep_ran = (
        "A7" in results and results["A7"] is not None
        and bool(results["A7"].get("position_sweep"))
    )

    if not curve_ran and not sweep_ran:
        print(
            "\n[!] All ablations skipped — no activation .npy files found."
            "\n    ablations_report.txt will NOT be saved (no data)."
            "\n    Generate activations first:"
            "\n      python scripts/cache_lie_detector_activations.py --filtered"
        )
        return

    report_text = "\n".join(sections)
    report_path = OUTPUT_DIR / "ablations_report.txt"
    report_path.write_text(report_text)
    print(f"\n[+] Saved: {report_path}")

    json_results = {}
    for k, v in results.items():
        if v is None:
            json_results[k] = None
        elif k == "A7":
            json_results[k] = v
        else:
            json_results[k] = {
                "description":   ABLATION_DESCRIPTIONS.get(k, ""),
                "fixed_holdout": v.get("fixed_holdout"),
                "test_size":     v.get("test_size"),
                "n_trials":      v.get("n_trials"),
                "c_values":      v.get("c_values"),
                "token_note":    v.get("token_note"),
                "auroc_by_n":    {str(n): d for n, d in v.get("auroc_by_n", {}).items()},
                "best_c_by_n":   {str(n): d for n, d in v.get("best_c_by_n", {}).items()},
            }
    json_results["paper_table_1"] = {str(n): a for n, a in PAPER_AUROC.items()}
    json_results["paper_C_values"] = PAPER_C_VALUES
    json_results["wide_C_values"]  = WIDE_C_VALUES

    json_path = OUTPUT_DIR / "ablations.json"
    json_path.write_text(json.dumps(json_results, indent=2))
    print(f"[+] Saved: {json_path}")

    print("\n" + report_text)


if __name__ == "__main__":
    main()
