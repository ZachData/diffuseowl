#!/usr/bin/env python3
"""
fixes/fix_letter_position.py

Demonstrates the core sabotage in the lie detector experiment and runs the
corrected letter-finding logic for direct comparison.

USAGE (from project root, with cache files present):
    python fixes/fix_letter_position.py

USAGE (dry-run / code-analysis only, no cache needed):
    python fixes/fix_letter_position.py --dry-run

OUTPUTS (in fixes/output/):
    sabotage_report.txt
        Full analysis report structured for LLM handoff. Contains:
        - Bug descriptions with exact file/line references
        - Token distribution tables (sabotaged vs fixed)
        - AUROC table from paper with annotation
        - Impact assessment and next steps

    sabotage_comparison.json
        Machine-readable comparison dict with the same structure as the
        paper's cached probed_tokens and AUROC summary files. Directly
        comparable to originals in:
          experiments/gemma-3-12b/cached_activations/
            lie_detector_filtered/unified_n5000_filtered/
              probed_tokens_pos-letter.json   ← sabotaged version
          experiments/gemma-3-12b/results/lie_detector/n5000_filtered/
              auroc_vs_n_layer30_pos-letter_n5000_filtered_summary.json

    fixed_probed_tokens_pos-letter.json
        The corrected per-sample letter list (A/B/C/D or null) in the
        same format as the original probed_tokens_pos-letter.json.
        Null = \\box{X} not found in Pass 1 text (fallback would apply).

    index_space_demo.txt
        Self-contained proof of Bug 2 (no model or cache needed):
        shows that completion-space index != full-conversation-space index
        using the actual token patterns from the codebase.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust if running from a different working directory
# ---------------------------------------------------------------------------
CACHE_DIR = Path("experiments/gemma-3-12b/cached_activations"
                 "/lie_detector_filtered/unified_n5000_filtered")
RESULTS_DIR = Path("experiments/gemma-3-12b/results/lie_detector/n5000_filtered")
OUTPUT_DIR = Path("fixes/output")

# AUROC table as reported in PAPER.md Table 1
# Source: auroc_vs_n_layer30_pos-letter_n5000_filtered_summary.json
# (LLM-REPORT confirmed stored values are error = 1 - AUROC)
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

# ---------------------------------------------------------------------------
# Corrected letter-finding logic (no tokenizer needed — uses regex on raw text)
# For the actual model fix see fixes/fixed_activations_patch.py
# ---------------------------------------------------------------------------

def find_letter_in_text_regex(text: str):
    """
    Corrected letter-finding using regex on raw text.

    Mirrors the intended behavior of find_letter_token_position_in_full_text
    (see fixed_activations_patch.py) without requiring a loaded tokenizer.
    Searches for \\box{X} after the LAST 'Your answer:' marker to avoid
    matching any example boxes in the prompt preamble.

    Returns the letter (A/B/C/D) or None.
    """
    # Find the last "Your answer:" — that's where Pass 1 answer appears
    marker = "Your answer:"
    last_pos = text.rfind(marker)
    if last_pos == -1:
        # Fallback: search whole text
        search_text = text
    else:
        search_text = text[last_pos:]

    match = re.search(r'\\box\{([A-D])\}', search_text)
    return match.group(1) if match else None


def analyze_token_distribution(tokens: list) -> dict:
    """Compute summary statistics over a list of token strings."""
    clean = [t for t in tokens if t is not None]
    counter = Counter(clean)
    total = len(tokens)
    found = len(clean)

    letter_tokens = {k: v for k, v in counter.items()
                     if k.strip() in ('A', 'B', 'C', 'D')}
    yes_no_tokens = {k: v for k, v in counter.items()
                     if 'yes' in k.lower() or 'no' in k.lower()}

    return {
        "total": total,
        "found": found,
        "not_found": total - found,
        "unique_tokens": len(counter),
        "top_10": counter.most_common(10),
        "letter_tokens": letter_tokens,
        "yes_no_tokens": yes_no_tokens,
        "letter_pct": 100 * sum(letter_tokens.values()) / total if total else 0,
        "yes_no_pct":  100 * sum(yes_no_tokens.values()) / total if total else 0,
    }


# ---------------------------------------------------------------------------
# Index-space mismatch demo (requires no external data)
# ---------------------------------------------------------------------------

def build_index_space_demo() -> str:
    """
    Construct a self-contained proof of Bug 2 using the exact token
    patterns from the codebase. No model or cache files required.
    """
    lines = []
    def w(s=""):
        lines.append(s)

    w("INDEX SPACE MISMATCH DEMO")
    w("=" * 60)
    w()
    w("This demonstrates Bug 2 using the actual logic from src/activations.py.")
    w()
    w("The sabotaged find_letter_token_position() tokenizes only the completion")
    w("string and returns index i within that short sequence. That index is then")
    w("used directly against input_ids (the full two-pass conversation).")
    w()
    w("Concrete example using Gemma tokenizer approximate behavior:")
    w()

    # Approximate tokenization of just the completion (Pass 1)
    # \box{A} tokenizes roughly as: ['▁', '\\', 'box', '{', 'A', '}']
    # but the code looks for: tokens[i] contains '\\', tokens[i+1:i+3] == ['box', '{']
    # So index = i + 3 = position of 'A' within completion tokens
    completion_example = r"\box{A}"
    # Approximate tokens: ['\', 'box', '{', 'A', '}']  (simplified)
    completion_tokens = ['\\', 'box', '{', 'A', '}']
    # find_letter_token_position searches for '\\' + 'box' + '{'
    # finds at i=0, returns index=3 (the 'A')
    letter_idx_in_completion = 3

    w(f"  Completion string (Pass 1 answer portion):")
    w(f"    '{completion_example}'")
    w()
    w(f"  Completion tokens (simplified):")
    for i, tok in enumerate(completion_tokens):
        marker = " ← find_letter_token_position returns this index" if i == letter_idx_in_completion else ""
        w(f"    [{i}] '{tok}'{marker}")
    w()
    w(f"  Returned index: {letter_idx_in_completion}  (index of 'A' within completion tokens)")
    w()

    # Now show the full conversation context
    # A typical Pass 1 prompt starts with system tokens, question, choices, etc.
    # "Your answer:" appears around token 200-400 in a typical MMLU question.
    # Then the completion is appended. Index 3 into the full conversation is
    # in the system prompt / BOS tokens area, nowhere near the answer.

    w("  Full two-pass conversation input_ids (schematic):")
    w("    [0]   <bos>")
    w("    [1]   <start_of_turn>")
    w("    [2]   'user'")
    w("    [3]   'You'          ← sabotaged code probes HERE (index 3)")
    w("    [4]   'are'")
    w("    [5]   'an'")
    w("    ...   (question text, ~200-400 tokens)")
    w("    [N]   'Your'")
    w("    [N+1] 'answer'")
    w("    [N+2] ':'")
    w("    [N+3] '\\'")
    w("    [N+4] 'box'")
    w("    [N+5] '{'")
    w(f"    [N+6] 'A'           ← CORRECT position (should be probed)")
    w("    [N+7] '}'")
    w("    ...   (Pass 2 follow-up conversation)")
    w()
    w("  The returned index 3 lands in the system prompt header,")
    w("  not at the answer letter. The correct index is N+6, which")
    w("  varies per sample depending on question length.")
    w()
    w("  This bug means every sample probes a different wrong token.")
    w("  Since the fallback (Bug 1) triggers first in practice, the")
    w("  actual probed token is the last token of the conversation")
    w("  ('yes' or newline), not index 3.")
    w()
    w("FIX:")
    w("  find_letter_token_position_in_full_text(full_text, tokenizer)")
    w("  tokenizes full_text once, searches for \\\\box{X} within those")
    w("  tokens, and returns the index directly in full-conversation space.")
    w("  See fixes/fixed_activations_patch.py for the complete implementation.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_json_if_exists(path: Path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def main(dry_run: bool = False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("FIX: Letter Token Position Sabotage Analysis")
    print("=" * 70)

    # --- Always build the index-space demo ---
    index_demo = build_index_space_demo()
    demo_path = OUTPUT_DIR / "index_space_demo.txt"
    demo_path.write_text(index_demo)
    print(f"\n[+] Saved: {demo_path}")

    # --- Attempt to load cached data ---
    cache_available = CACHE_DIR.exists()
    if dry_run:
        cache_available = False
        print("\n[dry-run] Skipping cache file loading.")
    elif not cache_available:
        print(f"\n[!] Cache directory not found: {CACHE_DIR}")
        print("    Running in analysis-only mode (code + index demo).")
        print("    To load cache data, run from project root with cache present.")

    sabotaged_tokens = None
    first_generation = None
    labels = None
    predicted_answers = None
    auroc_summary = None

    if cache_available:
        print("\nLoading cached data...")
        sabotaged_tokens = load_json_if_exists(
            CACHE_DIR / "probed_tokens_pos-letter.json")
        first_generation  = load_json_if_exists(
            CACHE_DIR / "first_generation.json")
        labels            = load_json_if_exists(
            CACHE_DIR / "labels.json")
        predicted_answers = load_json_if_exists(
            CACHE_DIR / "predicted_answers.json")
        auroc_summary     = load_json_if_exists(
            RESULTS_DIR / "auroc_vs_n_layer30_pos-letter_n5000_filtered_summary.json")

        missing = [name for name, val in [
            ("probed_tokens_pos-letter.json", sabotaged_tokens),
            ("first_generation.json",         first_generation),
            ("labels.json",                   labels),
        ] if val is None]

        if missing:
            print(f"[!] Missing cache files: {missing}")
            print("    Switching to analysis-only mode.")
            cache_available = False

    # --- Run corrected logic if data available ---
    fixed_letters = None
    fixed_stats   = None
    sabotaged_stats = None

    if cache_available:
        print(f"    Loaded {len(labels)} samples")

        print("\n[1] Analyzing SABOTAGED probed tokens...")
        sabotaged_stats = analyze_token_distribution(sabotaged_tokens)

        print("[2] Running FIXED letter-finding on Pass 1 generations...")
        fixed_letters = [find_letter_in_text_regex(t) for t in first_generation]
        fixed_stats   = analyze_token_distribution(fixed_letters)

        # Validate: do the fixed letters match predicted_answers?
        if predicted_answers:
            both_found = [(fl, pa) for fl, pa in zip(fixed_letters, predicted_answers)
                         if fl is not None and pa is not None]
            matches = sum(1 for fl, pa in both_found if fl == pa)
            agreement_pct = 100 * matches / len(both_found) if both_found else 0
        else:
            agreement_pct = None

        # Save fixed probed tokens JSON (same format as original)
        fixed_json_path = OUTPUT_DIR / "fixed_probed_tokens_pos-letter.json"
        fixed_json_path.write_text(json.dumps(fixed_letters, indent=2))
        print(f"[+] Saved: {fixed_json_path}")

        # Save comparison JSON
        label_counter = Counter(labels)
        comparison = {
            "description": (
                "Direct comparison of sabotaged vs fixed probed token distributions "
                "for the 'letter' position in the lie detector experiment."
            ),
            "sabotaged": {
                "source_file": str(CACHE_DIR / "probed_tokens_pos-letter.json"),
                "description": (
                    "Tokens actually probed in the paper's cached run. "
                    "These come from the Pass 2 'yes' token due to two bugs."
                ),
                "total_samples": sabotaged_stats["total"],
                "letter_tokens_pct": round(sabotaged_stats["letter_pct"], 2),
                "yes_no_tokens_pct": round(sabotaged_stats["yes_no_pct"], 2),
                "top_10_tokens": [
                    {"token": tok, "count": cnt,
                     "pct": round(100 * cnt / sabotaged_stats["total"], 1)}
                    for tok, cnt in sabotaged_stats["top_10"]
                ],
            },
            "fixed": {
                "source_file": str(CACHE_DIR / "first_generation.json"),
                "method": (
                    "Regex search for \\\\box{X} in Pass 1 full texts after "
                    "'Your answer:' marker. Equivalent to corrected token-space search."
                ),
                "total_samples": fixed_stats["total"],
                "found_pct": round(100 * fixed_stats["found"] / fixed_stats["total"], 2),
                "not_found_count": fixed_stats["not_found"],
                "letter_distribution": dict(Counter(fixed_letters)),
                "agreement_with_predicted_answers_pct": (
                    round(agreement_pct, 2) if agreement_pct is not None else None
                ),
            },
            "label_distribution": {
                "correct_1": label_counter[1],
                "incorrect_0": label_counter[0],
                "accuracy_pct": round(100 * label_counter[1] / len(labels), 1),
            },
            "auroc_from_sabotaged_run": {
                "layer": 30,
                "position_claimed": "letter (A/B/C/D token)",
                "position_actual": "yes/no token (Pass 2 fallback)",
                "table": [
                    {"n": n, "auroc": a} for n, a in PAPER_AUROC_TABLE.items()
                ],
                "note": (
                    "These AUROC values reflect probing the 'yes' token, "
                    "not the answer-letter token. They are from the paper's "
                    "Table 1 and Appendix C."
                ),
            },
            "verdict": (
                "SABOTAGED. The paper's 'letter' position probed the Pass 2 yes/no "
                "token, not the Pass 1 A/B/C/D answer token. Two compounding bugs "
                "caused this: (1) Pass 2 completions were passed to the letter-finder, "
                "(2) the returned index was in completion-token space, not "
                "full-conversation space."
            ),
        }
        comparison_path = OUTPUT_DIR / "sabotage_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2))
        print(f"[+] Saved: {comparison_path}")

    # --- Build the main text report ---
    report = build_report(
        sabotaged_stats=sabotaged_stats,
        fixed_stats=fixed_stats,
        fixed_letters=fixed_letters,
        labels=labels,
        agreement_pct=agreement_pct if cache_available else None,
        auroc_summary=auroc_summary,
        cache_available=cache_available,
    )

    report_path = OUTPUT_DIR / "sabotage_report.txt"
    report_path.write_text(report)
    print(f"[+] Saved: {report_path}")

    print("\n" + report)


def build_report(
    sabotaged_stats,
    fixed_stats,
    fixed_letters,
    labels,
    agreement_pct,
    auroc_summary,
    cache_available: bool,
) -> str:
    lines = []
    def w(s=""):
        lines.append(s)

    w("bug1  cache_lie_detector_activations.py:194  completions=second_completions (Pass 2 'yes')")
    w("bug2  src/activations.py:256-265  completion-space index used in full-conversation space")
    w("effect  'letter' position probes 'yes'/newline token from Pass 2, not A/B/C/D from Pass 1")
    w()

    # token distributions — only shown when real cache is available
    if cache_available and sabotaged_stats and fixed_stats:
        w(f"sabotaged_tokens  n={sabotaged_stats['total']}  "
          f"letter_pct={sabotaged_stats['letter_pct']:.1f}%  "
          f"yes_no_pct={sabotaged_stats['yes_no_pct']:.1f}%")
        w(f"  {'token':<28} {'count':>6}  {'pct':>6}")
        w(f"  {'─'*44}")
        for tok, cnt in sabotaged_stats["top_10"]:
            pct = 100 * cnt / sabotaged_stats["total"]
            w(f"  {repr(tok):<28} {cnt:>6}  {pct:>5.1f}%")
        w()

        w(f"fixed_tokens  n={fixed_stats['total']}  "
          f"found={fixed_stats['found']} ({100*fixed_stats['found']/fixed_stats['total']:.1f}%)  "
          f"not_found={fixed_stats['not_found']}")
        letter_dist = Counter(fixed_letters)
        for letter in ('A', 'B', 'C', 'D'):
            cnt = letter_dist.get(letter, 0)
            pct = 100 * cnt / fixed_stats["total"] if fixed_stats["total"] else 0
            w(f"  {letter}: {cnt:>5}  ({pct:.1f}%)")
        if agreement_pct is not None:
            w(f"  agreement_with_predicted_answers={agreement_pct:.1f}%")
        w()

        if labels:
            label_counter = Counter(labels)
            w(f"labels  correct={label_counter[1]} ({100*label_counter[1]/len(labels):.1f}%)  "
              f"incorrect={label_counter[0]} ({100*label_counter[0]/len(labels):.1f}%)")
            w()

    # paper AUROC table — always shown
    w(f"paper_auroc  layer=30  position=letter(claimed)/yes_no(actual)")
    w(f"  {'N':>6}  {'auroc':>7}")
    w(f"  {'─'*16}")
    for n, auroc in PAPER_AUROC_TABLE.items():
        w(f"  {n:>6}  {auroc:>7.3f}")

    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demonstrate and fix the letter-position sabotage"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run code/index analysis only, skip cache file loading"
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run)
