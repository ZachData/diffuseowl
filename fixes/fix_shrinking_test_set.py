#!/usr/bin/env python3
"""
fixes/fix_shrinking_test_set.py

Demonstrates the shrinking test set bug in probes.py:1638-1643 and runs both
the sabotaged and corrected evaluation side by side on synthetic data.

WHAT WAS WRONG:
  probes.py, measure_auroc_vs_training_size(), lines 1638-1643:

    for n in n_values:
        for trial in range(n_trials):
            np.random.seed(random_state + trial)
            idx = np.random.permutation(len(activations))
            X_train = activations[idx[:n]]
            X_test  = activations[idx[n:]]   # ← shrinks as n grows

  For each training size N, everything after the first N samples becomes the
  test set. As N increases toward the total dataset size:
    - Test set shrinks from ~4484 samples (N=16) to ~0 samples (N=4500)
    - Each N gets a different, incomparable test distribution
    - AUROC at large N is estimated from a handful of samples (high variance)
    - The "steady improvement" curve in Table 1 conflates genuine learning
      signal with the artifact of a shrinking, shifting test set

  The paper (§3) presents these as directly comparable data points on a
  scaling curve. They are not.

WHAT THE FIX DOES:
  Split once into a fixed train_pool (80%) and a fixed test set (20%).
  For each N, subsample N from train_pool. Test set is identical across
  all N values.

    total_n = len(activations)
    test_size = int(total_n * TEST_FRACTION)

    np.random.seed(random_state)
    pool_idx = np.random.permutation(total_n)
    test_idx   = pool_idx[:test_size]
    train_pool = pool_idx[test_size:]

    for n in n_values:
        for trial in range(n_trials):
            np.random.seed(random_state + trial)
            chosen = np.random.choice(train_pool, size=n, replace=False)
            X_train = activations[chosen]
            X_test  = activations[test_idx]   # ← constant

HOW TO APPLY:
  In src/probes.py, replace measure_auroc_vs_training_size() with
  measure_auroc_vs_training_size_fixed() (below), or apply the
  targeted diff printed in fixes/output/shrinking_test_set_report.txt.

USAGE:
  python fixes/fix_shrinking_test_set.py              # synthetic data demo
  python fixes/fix_shrinking_test_set.py --cache-dir experiments/gemma-3-12b/...
                                                      # run on real activations
"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

OUTPUT_DIR = Path("fixes/output")

# Match the paper's N values from Table 1
PAPER_N_VALUES = [16, 32, 64, 128, 256, 512, 1024, 2048, 4500]

# Paper AUROC table for reference
PAPER_AUROC_TABLE = {
    16:   0.621,
    32:   0.764,
    64:   0.828,
    128:  0.888,
    256:  0.909,
    512:  0.936,
    1024: 0.948,
    2048: 0.956,
    4500: 0.966,
}

TEST_FRACTION = 0.20   # fixed holdout fraction for the corrected version
N_TRIALS      = 3      # trials per N (paper used 1; we use 3 to show variance)
RANDOM_STATE  = 42


# ---------------------------------------------------------------------------
# Sabotaged version — exactly mirrors probes.py:1633-1663
# ---------------------------------------------------------------------------

def measure_auroc_sabotaged(activations, labels, n_values, n_trials, random_state):
    """
    Exact replica of the sabotaged measure_auroc_vs_training_size from probes.py.
    Test set = idx[n:], which shrinks as n grows.
    """
    test_results  = {n: [] for n in n_values}
    test_sizes    = {n: [] for n in n_values}

    for n in n_values:
        for trial in range(n_trials):
            np.random.seed(random_state + trial)
            idx = np.random.permutation(len(activations))

            X_train = activations[idx[:n]]
            y_train = labels[idx[:n]]
            X_test  = activations[idx[n:]]   # ← shrinks
            y_test  = labels[idx[n:]]

            test_sizes[n].append(len(X_test))

            if len(X_test) == 0 or len(np.unique(y_train)) < 2:
                continue

            clf = LogisticRegression(
                C=1e-5, max_iter=1000, solver='lbfgs', random_state=random_state
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) < 2:
                continue
            auroc = roc_auc_score(y_test, proba)
            test_results[n].append(auroc)

    return test_results, test_sizes


# ---------------------------------------------------------------------------
# Fixed version
# ---------------------------------------------------------------------------

def measure_auroc_fixed(activations, labels, n_values, n_trials, random_state,
                        test_fraction=TEST_FRACTION):
    """
    Corrected measure_auroc_vs_training_size.
    Splits once into fixed test set and train pool; subsamples N from pool per trial.
    """
    total_n   = len(activations)
    test_size = int(total_n * test_fraction)

    # One-time split
    np.random.seed(random_state)
    pool_idx   = np.random.permutation(total_n)
    test_idx   = pool_idx[:test_size]
    train_pool = pool_idx[test_size:]

    X_test = activations[test_idx]
    y_test = labels[test_idx]

    test_results = {n: [] for n in n_values}

    for n in n_values:
        if n > len(train_pool):
            # More training samples requested than pool has; skip
            continue
        for trial in range(n_trials):
            np.random.seed(random_state + trial)
            chosen  = np.random.choice(train_pool, size=n, replace=False)
            X_train = activations[chosen]
            y_train = labels[chosen]

            if len(np.unique(y_train)) < 2:
                continue

            clf = LogisticRegression(
                C=1e-5, max_iter=1000, solver='lbfgs', random_state=random_state
            )
            clf.fit(X_train, y_train)
            proba = clf.predict_proba(X_test)[:, 1]
            if len(np.unique(y_test)) < 2:
                continue
            auroc = roc_auc_score(y_test, proba)
            test_results[n].append(auroc)

    return test_results, test_size


# ---------------------------------------------------------------------------
# Data generation / loading
# ---------------------------------------------------------------------------

def make_synthetic_data(n_samples=5000, n_features=3072, signal_strength=0.4,
                        seed=0):
    """
    Generate synthetic activations that mimic the lie detector setup:
    roughly balanced classes, moderate linear separability.
    """
    rng = np.random.default_rng(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    # Low-rank signal: a single direction with added noise
    signal_dir = rng.standard_normal(n_features)
    signal_dir /= np.linalg.norm(signal_dir)

    X_pos = rng.standard_normal((n_pos, n_features)) + signal_strength * signal_dir
    X_neg = rng.standard_normal((n_neg, n_features)) - signal_strength * signal_dir
    X = np.vstack([X_pos, X_neg])
    y = np.array([1] * n_pos + [0] * n_neg)

    # Shuffle
    idx = rng.permutation(n_samples)
    return X[idx].astype(np.float32), y[idx]


def load_real_data(cache_dir: str, layer: int = 30, position: str = "letter"):
    """
    Load real data from the experiment cache.

    Strategy:
      1. If activation .npy exists → use it directly (full fidelity)
      2. If only labels.json exists → generate a random projection as stand-in
         activations. This still demonstrates the test-set-size bug correctly
         because the bug is in the *evaluation protocol*, not the activations.
      3. Neither → return None, None, error message
    """
    cache = Path(cache_dir)
    act_file    = cache / f"activations_layer{layer:02d}_pos-{position}.npy"
    labels_file = cache / "labels.json"

    if not labels_file.exists():
        return None, None, f"Labels not found: {labels_file}"

    with open(labels_file) as f:
        labels = np.array(json.load(f))

    if act_file.exists():
        activations = np.load(act_file)
        source = f"Real activations: layer={layer} pos={position} n={len(labels)}"
        return activations, labels, source

    # No activation .npy — generate a reproducible random projection of the
    # correct shape so the evaluation-protocol comparison is still meaningful.
    n = len(labels)
    rng = np.random.default_rng(42)
    # Inject a small label-correlated signal so AUROC > 0.5
    signal_dim = 32
    features   = 128   # small — fast, still illustrates the bug
    noise      = rng.standard_normal((n, features)).astype(np.float32)
    signal_dir = rng.standard_normal(features).astype(np.float32)
    signal_dir /= np.linalg.norm(signal_dir)
    signal_strength = 0.3
    noise[labels == 1] += signal_strength * signal_dir
    noise[labels == 0] -= signal_strength * signal_dir
    activations = noise
    source = (
        f"Real labels + random projection (activation .npy not yet generated).\n"
        f"  n={n}  features={features}  "
        f"Run cache_lie_detector_activations.py --filtered for real activations."
    )
    print(f"  [info] Activation .npy not found; using real labels + random projection.")
    print(f"         The test-set-size comparison is still valid (bug is in evaluation,")
    print(f"         not activations). Re-run after generating activations for full results.")
    return activations, labels, source


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

def build_report(
    sabotaged_results, sabotaged_test_sizes,
    fixed_results, fixed_test_size,
    n_values, n_trials, data_source, total_n,
    paper_auroc=None,
) -> str:
    lines = []
    def w(s=""):
        lines.append(s)

    source_line = data_source.split("\n")[0].strip()
    w(f"bug   src/probes.py:1638  X_test=idx[n:] shrinks with N")
    w(f"data  {source_line}")
    w(f"      n={total_n}  fixed_test={fixed_test_size} ({100*fixed_test_size/total_n:.0f}%)  "
      f"train_pool={total_n-fixed_test_size}  trials={n_trials}")
    w()

    # test set sizes
    w(f"  {'N':>6}  {'sab_test_n':>12}  {'fix_test_n':>12}")
    w(f"  {'─'*34}")
    for n in n_values:
        sz = sabotaged_test_sizes.get(n, [])
        sab_sz = str(int(np.mean(sz))) if sz else "N/A"
        w(f"  {n:>6}  {sab_sz:>12}  {fixed_test_size:>12}")
    w()

    # AUROC table
    has_paper = bool(paper_auroc)
    header = f"  {'N':>6}  {'sab_auroc(mean±std)':>20}  {'fix_auroc(mean±std)':>20}"
    if has_paper:
        header += f"  {'paper':>7}"
    w(header)
    w(f"  {'─'*(len(header)-2)}")
    for n in n_values:
        sab = sabotaged_results.get(n, [])
        fix = fixed_results.get(n, [])
        sab_str = f"{np.mean(sab):.3f}±{np.std(sab):.3f}" if sab else "N/A"
        fix_str = f"{np.mean(fix):.3f}±{np.std(fix):.3f}" if fix else "N/A"
        row = f"  {n:>6}  {sab_str:>20}  {fix_str:>20}"
        if has_paper:
            pa = paper_auroc.get(n)
            row += f"  {pa:>7.3f}" if pa else "      N/A"
        w(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cache_dir=None, layer=30, position="letter"):
    if not SKLEARN_AVAILABLE:
        print("scikit-learn not found. Install with: pip install scikit-learn")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    activations = labels = None
    data_source = None

    if cache_dir:
        result = load_real_data(cache_dir, layer, position)
        activations, labels, source_or_err = result
        if activations is None:
            print(f"[!] Could not load data: {source_or_err}")
            print("    Falling back to fully synthetic data.")
        else:
            data_source = source_or_err
            print(f"[+] Loaded data: shape={activations.shape}")

    if activations is None:
        print("[*] Generating synthetic data (5000 samples, 3072 features)...")
        activations, labels = make_synthetic_data(
            n_samples=5000, n_features=3072, signal_strength=0.4
        )
        data_source = (
            "Synthetic activations (5000 samples, 3072 features, "
            "signal_strength=0.4).\n"
            "  Designed to approximate the lie detector activation geometry."
        )

    total_n = len(activations)
    print(f"[*] Total samples: {total_n}, label balance: "
          f"{int(np.sum(labels))} pos / {int(total_n - np.sum(labels))} neg")

    # Filter n_values to what is feasible
    n_values = [n for n in PAPER_N_VALUES if n < total_n]
    print(f"[*] N values: {n_values}")

    # Run sabotaged version
    print("\n[1] Running SABOTAGED evaluation (shrinking test set)...")
    sab_results, sab_test_sizes = measure_auroc_sabotaged(
        activations, labels, n_values, N_TRIALS, RANDOM_STATE
    )

    # Run fixed version
    print("[2] Running FIXED evaluation (constant test set)...")
    fix_results, fix_test_size = measure_auroc_fixed(
        activations, labels, n_values, N_TRIALS, RANDOM_STATE
    )

    # Determine whether to show paper AUROC (only meaningful with real data at letter pos)
    show_paper = (cache_dir is not None and position == "letter" and layer == 30)
    paper_auroc = PAPER_AUROC_TABLE if show_paper else None

    # Build and save report
    report = build_report(
        sab_results, sab_test_sizes,
        fix_results, fix_test_size,
        n_values, N_TRIALS, data_source, total_n,
        paper_auroc=paper_auroc,
    )

    report_path = OUTPUT_DIR / "shrinking_test_set_report.txt"
    report_path.write_text(report)
    print(f"\n[+] Saved: {report_path}")
    print("\n" + report)

    # Save comparison JSON (same structure as original summary JSON from probe_analysis.py)
    comparison = {
        "description": (
            "Comparison of sabotaged (shrinking test set) vs fixed (constant test set) "
            "AUROC-vs-N evaluation. Mirrors the format of "
            "auroc_vs_n_layer30_pos-letter_n5000_filtered_summary.json."
        ),
        "bug": {
            "file": "src/probes.py",
            "function": "measure_auroc_vs_training_size",
            "lines": "1638-1643",
            "description": "X_test = activations[idx[n:]] shrinks as n grows",
        },
        "data_source": data_source,
        "total_samples": total_n,
        "n_trials": N_TRIALS,
        "sabotaged": {
            "description": "Test set = idx[n:], changes and shrinks with every N",
            "test_sizes_by_n": {
                str(n): int(np.mean(v)) if v else None
                for n, v in sab_test_sizes.items()
            },
            "auroc_by_n": {
                str(n): {
                    "mean": round(float(np.mean(v)), 4) if v else None,
                    "std":  round(float(np.std(v)),  4) if v else None,
                    "trials": [round(float(x), 4) for x in v],
                }
                for n, v in sab_results.items()
            },
        },
        "fixed": {
            "description": (
                f"Fixed test set: {fix_test_size} samples ({100*fix_test_size/total_n:.0f}%). "
                "Same test set for every N. Train pool subsampled per trial."
            ),
            "test_size": fix_test_size,
            "auroc_by_n": {
                str(n): {
                    "mean": round(float(np.mean(v)), 4) if v else None,
                    "std":  round(float(np.std(v)),  4) if v else None,
                    "trials": [round(float(x), 4) for x in v],
                }
                for n, v in fix_results.items()
            },
        },
        "paper_auroc_table_1": (
            {str(n): a for n, a in PAPER_AUROC_TABLE.items()}
            if show_paper else
            "N/A (only shown when evaluating real layer-30 letter-position activations)"
        ),
    }

    comparison_path = OUTPUT_DIR / "shrinking_test_set_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))
    print(f"[+] Saved: {comparison_path}")


# ---------------------------------------------------------------------------
# Drop-in replacement for src/probes.py (importable by other scripts)
# ---------------------------------------------------------------------------

def measure_auroc_vs_training_size_fixed(
    activations: np.ndarray,
    labels: np.ndarray,
    n_values=None,
    n_trials: int = 3,
    max_iter: int = 1000,
    random_state: int = 42,
    test_fraction: float = TEST_FRACTION,
):
    """
    Fixed replacement for measure_auroc_vs_training_size in src/probes.py.

    Uses a constant held-out test set for all N values.
    Signature matches the original; callers require no changes.

    Args:
        activations: np.ndarray (n_samples, hidden_dim)
        labels: np.ndarray (n_samples,) binary
        n_values: list of training sizes; auto-computed if None
        n_trials: trials per N (paper used 1; recommend >= 3)
        max_iter: logistic regression max iterations
        random_state: base random seed
        test_fraction: fraction of data held out as fixed test set

    Returns:
        (test_results, train_results, cv_info) — same structure as original
    """
    from sklearn.metrics import roc_auc_score

    if n_values is None:
        import math
        high = int(math.floor(math.log2(len(activations)))) - 1
        n_values = [2 ** i for i in range(4, high + 1)]
        top = int(len(activations) * 0.9 * (1 - test_fraction))
        n_values.append(top)
        n_values = sorted(set(n_values))

    # One-time split
    total_n   = len(activations)
    test_size = int(total_n * test_fraction)
    np.random.seed(random_state)
    pool_idx   = np.random.permutation(total_n)
    test_idx   = pool_idx[:test_size]
    train_pool = pool_idx[test_size:]

    X_test = activations[test_idx]
    y_test = labels[test_idx]

    test_results  = {n: [] for n in n_values}
    train_results = {n: [] for n in n_values}
    cv_info       = {}

    for n in n_values:
        print(f"Training linear probes with {n} examples (fixed test set, n={test_size})")
        if n > len(train_pool):
            print(f"  Skipping N={n}: exceeds train pool ({len(train_pool)})")
            continue

        for trial in range(n_trials):
            np.random.seed(random_state + trial)
            chosen  = np.random.choice(train_pool, size=n, replace=False)
            X_train = activations[chosen]
            y_train = labels[chosen]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            # Import here to keep this module usable without the full project
            try:
                from src.probes import train_linear_probe
                auroc_test, clf, _, _, _ = train_linear_probe(
                    X_train, y_train, X_test, y_test, max_iter, random_state
                )
            except ImportError:
                clf = LogisticRegression(
                    C=1e-5, max_iter=max_iter, solver='lbfgs',
                    random_state=random_state
                )
                clf.fit(X_train, y_train)
                proba = clf.predict_proba(X_test)[:, 1]
                auroc_test = roc_auc_score(y_test, proba)

            test_results[n].append(1 - auroc_test)

            proba_train = clf.predict_proba(X_train)[:, 1]
            auroc_train = roc_auc_score(y_train, proba_train)
            train_results[n].append(1 - auroc_train)

            if hasattr(clf, 'best_alpha_') and clf.best_alpha_ is not None:
                cv_info[(n, trial)] = {
                    'best_alpha': clf.best_alpha_,
                    'cv_results': clf.cv_results_,
                }

    return test_results, train_results, cv_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate and fix the shrinking test set bug"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help=(
            "Path to unified activation cache dir "
            "(e.g. experiments/gemma-3-12b/cached_activations/"
            "lie_detector_filtered/unified_n5000_filtered). "
            "Uses synthetic data if not provided."
        )
    )
    parser.add_argument("--layer",    type=int, default=30)
    parser.add_argument("--position", type=str, default="letter")
    args = parser.parse_args()
    main(cache_dir=args.cache_dir, layer=args.layer, position=args.position)
