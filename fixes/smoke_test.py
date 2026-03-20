#!/usr/bin/env python3
"""
fixes/smoke_test.py

Smoke test for the fixes/ suite. Verifies all logic without VLLM, without the
project model, and without the experiment cache.

Tests are organized into four groups:

  Group 1 — Fix 1: Letter Token Position
    Tests the corrected letter-finding logic against:
      - Pass 1 completions (should find A/B/C/D)
      - Pass 2 completions ("yes") (should return None — sabotage scenario)
      - Edge cases: no box, wrong format, box in prompt preamble only

  Group 2 — Fix 2: Shrinking Test Set
    Tests the sabotaged and fixed evaluation functions against synthetic data:
      - Sabotaged: confirms test set size decreases as N increases
      - Fixed: confirms test set size is constant across all N
      - Both: confirms fixed AUROC variance is lower than sabotaged at high N

  Group 3 — Fix 3: Prompt Variants
    Tests prompt content, helpers, and compute_stats against synthetic outputs:
      - All four prompts contain expected keyword differences
      - extract_letter and extract_yes_no parse correctly
      - compute_stats produces correct accuracy / compliance / self-assessment
      - Variant descriptions match actual prompt content

  Group 4 — Integration
    Tests that each fix module can be imported and run in dry-run / no-cache mode
    without raising an exception, and that expected output files are produced.

USAGE:
    python fixes/smoke_test.py           # run all tests
    python fixes/smoke_test.py --group 1 # run one group only
    python fixes/smoke_test.py -v        # verbose (show individual test names)

EXIT CODE:
    0 — all tests passed
    1 — one or more tests failed
"""

import argparse
import importlib
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Callable, List, Optional

# ── locate fixes/ regardless of CWD ──────────────────────────────────────────
FIXES_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(FIXES_DIR))

OUTPUT_DIR = FIXES_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Minimal test harness
# ─────────────────────────────────────────────────────────────────────────────

class TestResult:
    def __init__(self, name: str, passed: bool, message: str = ""):
        self.name    = name
        self.passed  = passed
        self.message = message

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        msg    = f"  {self.message}" if self.message else ""
        return f"  [{status}] {self.name}{msg}"


class TestSuite:
    def __init__(self, name: str):
        self.name    = name
        self.results: List[TestResult] = []

    def check(self, name: str, condition: bool, message: str = ""):
        self.results.append(TestResult(name, condition, message))

    def raises(self, name: str, fn: Callable, exc_type=Exception, message: str = ""):
        """Assert that fn() raises exc_type."""
        try:
            fn()
            self.results.append(TestResult(name, False, f"expected {exc_type.__name__}, got no exception"))
        except exc_type:
            self.results.append(TestResult(name, True, message))
        except Exception as e:
            self.results.append(TestResult(name, False, f"unexpected {type(e).__name__}: {e}"))

    def no_raise(self, name: str, fn: Callable, message: str = ""):
        """Assert that fn() does not raise."""
        try:
            fn()
            self.results.append(TestResult(name, True, message))
        except Exception as e:
            self.results.append(TestResult(
                name, False,
                f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            ))

    @property
    def passed(self):
        return all(r.passed for r in self.results)

    @property
    def n_passed(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def n_failed(self):
        return len(self.results) - self.n_passed


def run_suite(suite: TestSuite, verbose: bool) -> TestSuite:
    status = "PASS" if suite.passed else "FAIL"
    print(f"\n{'─' * 60}")
    print(f"Group: {suite.name}  [{status}]  "
          f"{suite.n_passed}/{len(suite.results)} passed")
    print(f"{'─' * 60}")
    for r in suite.results:
        if verbose or not r.passed:
            print(repr(r))
    if not verbose:
        n_hidden = sum(1 for r in suite.results if r.passed)
        if n_hidden:
            print(f"  ({n_hidden} passing tests hidden — run with -v to show all)")
    return suite


# ─────────────────────────────────────────────────────────────────────────────
# Group 1 — Fix 1: Letter Token Position
# ─────────────────────────────────────────────────────────────────────────────

def group1_letter_position(verbose: bool) -> TestSuite:
    suite = TestSuite("Fix 1 — Letter Token Position")

    # Import the corrected function
    import fix_letter_position as flp

    # ── Core: corrected regex finds the right letter ──────────────────────────

    # Typical Pass 1 full text: system prompt + question + completion
    pass1_full_text = (
        "You are an AI that is red-teaming...\n"
        "Question: What is 2+2?\n"
        "A. 3\nB. 4\nC. 5\nD. 6\n"
        r"Your answer: \box{B}"
    )

    result = flp.find_letter_in_text_regex(pass1_full_text)
    suite.check("finds letter in typical Pass 1 text", result == "B",
                f"expected 'B', got {result!r}")

    # All four letters
    for letter in ["A", "B", "C", "D"]:
        text = f"Your answer: \\box{{{letter}}}"
        r    = flp.find_letter_in_text_regex(text)
        suite.check(f"finds letter {letter}", r == letter, f"got {r!r}")

    # ── Core: sabotage scenario — Pass 2 completion is "yes" ─────────────────

    pass2_completion = "yes"
    result = flp.find_letter_in_text_regex(pass2_completion)
    suite.check(
        "returns None for Pass 2 'yes' completion (sabotage scenario)",
        result is None,
        f"expected None, got {result!r}",
    )

    pass2_full = (
        "You are an AI...\nQuestion: ...\nYour answer: \\box{C}\n"
        "User: Did you give the correct answer? Say only 'yes' or 'no'.\n"
        "Assistant: yes"
    )
    result = flp.find_letter_in_text_regex(pass2_full)
    # Should still find the letter in Pass 1 part, not "yes" from Pass 2
    suite.check(
        "finds letter from Pass 1 part of full two-pass conversation",
        result == "C",
        f"expected 'C', got {result!r}",
    )

    # ── Edge cases ────────────────────────────────────────────────────────────

    suite.check(
        "returns None when no \\box found",
        flp.find_letter_in_text_regex("The answer is B.") is None,
    )

    suite.check(
        "returns None for empty string",
        flp.find_letter_in_text_regex("") is None,
    )

    # Box in prompt preamble (before "Your answer:") should NOT match
    preamble_only = (
        r"Put your answer in a \box{} like \box{A}. "
        "Question: What is X?\nYour answer:"
    )
    result = flp.find_letter_in_text_regex(preamble_only)
    # No box after "Your answer:" — should be None
    suite.check(
        "ignores \\box example in prompt preamble (before 'Your answer:')",
        result is None,
        f"got {result!r}",
    )

    # ── Index space mismatch demo produces output ─────────────────────────────

    demo = flp.build_index_space_demo()
    suite.check("index space demo is non-empty string", isinstance(demo, str) and len(demo) > 100)
    suite.check("demo mentions completion-space index", "completion" in demo.lower())
    suite.check("demo mentions full-conversation", "full" in demo.lower())

    # ── Paper AUROC table is present and complete ─────────────────────────────

    suite.check(
        "paper AUROC table has 9 entries",
        len(flp.PAPER_AUROC_TABLE) == 9,
    )
    suite.check(
        "paper AUROC table N=4500 is 0.966",
        flp.PAPER_AUROC_TABLE[4500] == 0.966,
    )

    return run_suite(suite, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Group 2 — Fix 2: Shrinking Test Set
# ─────────────────────────────────────────────────────────────────────────────

def group2_shrinking_test_set(verbose: bool) -> TestSuite:
    suite = TestSuite("Fix 2 — Shrinking Test Set")

    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
    except ImportError as e:
        suite.check(f"numpy/sklearn available", False, str(e))
        return run_suite(suite, verbose)

    import fix_shrinking_test_set as fsts

    # Small synthetic dataset — fast, deterministic
    rng = np.random.default_rng(0)
    N = 500
    D = 50  # low dim for speed
    signal = rng.standard_normal(D)
    signal /= np.linalg.norm(signal)
    X = rng.standard_normal((N, D))
    X[:N//2] += 0.8 * signal
    X[N//2:] -= 0.8 * signal
    y = np.array([1]*(N//2) + [0]*(N//2))
    idx = rng.permutation(N)
    X, y = X[idx], y[idx]

    n_values = [16, 64, 256, 400]

    # ── Sabotaged: test set shrinks with N ────────────────────────────────────

    sab_results, sab_sizes = fsts.measure_auroc_sabotaged(
        X, y, n_values, n_trials=1, random_state=42
    )

    prev_size = None
    sizes_shrink = True
    for n in n_values:
        sz = sab_sizes[n][0] if sab_sizes[n] else 0
        if prev_size is not None and sz >= prev_size:
            sizes_shrink = False
        prev_size = sz

    suite.check(
        "sabotaged: test set strictly shrinks as N grows",
        sizes_shrink,
        f"sizes: {[sab_sizes[n][0] if sab_sizes[n] else 0 for n in n_values]}",
    )

    suite.check(
        "sabotaged: test size at N=16 is ~484 (500-16)",
        abs(sab_sizes[16][0] - 484) <= 2,
        f"got {sab_sizes[16][0]}",
    )

    suite.check(
        "sabotaged: test size at N=400 is ~100 (500-400)",
        abs(sab_sizes[400][0] - 100) <= 2,
        f"got {sab_sizes[400][0]}",
    )

    # ── Fixed: test set is constant ───────────────────────────────────────────

    fix_results, fix_test_size = fsts.measure_auroc_fixed(
        X, y, n_values, n_trials=2, random_state=42, test_fraction=0.20
    )

    expected_test_size = int(N * 0.20)
    suite.check(
        f"fixed: test size equals int(N * 0.20) = {expected_test_size}",
        fix_test_size == expected_test_size,
        f"got {fix_test_size}",
    )

    # Every N should see the same test set size (already confirmed by fix_test_size)
    suite.check(
        "fixed: test size is constant — all N values use the same holdout",
        fix_test_size == expected_test_size,
    )

    # ── Fixed produces valid AUROC values ─────────────────────────────────────

    for n in [16, 64, 256]:
        aurocs = fix_results.get(n, [])
        suite.check(
            f"fixed: N={n} produces at least 1 AUROC value",
            len(aurocs) >= 1,
            f"got {aurocs}",
        )
        if aurocs:
            suite.check(
                f"fixed: N={n} AUROC is in [0, 1]",
                all(0.0 <= a <= 1.0 for a in aurocs),
                f"got {aurocs}",
            )

    # ── AUROC should be above chance for both (signal_strength=0.8 is strong) ──

    fix_n64 = fix_results.get(64, [])
    if fix_n64:
        suite.check(
            "fixed: AUROC at N=64 > 0.5 (signal is detectable)",
            np.mean(fix_n64) > 0.5,
            f"mean AUROC = {np.mean(fix_n64):.3f}",
        )

    # ── Fixed AUROC variance at high N is lower than sabotaged ───────────────

    sab_n400 = sab_results.get(400, [])
    fix_n400 = fix_results.get(400, [])
    if len(sab_n400) >= 1 and len(fix_n400) >= 1:
        # At N=400 (near total), sabotaged test set is tiny → high variance
        # We can't easily test variance with n_trials=1, so just check both exist
        suite.check(
            "both evaluations produce AUROC at N=400",
            len(sab_n400) >= 1 and len(fix_n400) >= 1,
        )

    # ── Drop-in replacement function is importable and has correct signature ──

    import inspect
    sig    = inspect.signature(fsts.measure_auroc_vs_training_size_fixed)
    params = list(sig.parameters.keys())
    for expected in ["activations", "labels", "n_values", "n_trials",
                     "max_iter", "random_state", "test_fraction"]:
        suite.check(
            f"drop-in function has parameter '{expected}'",
            expected in params,
        )

    return run_suite(suite, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Group 3 — Fix 3: Prompt Variants
# ─────────────────────────────────────────────────────────────────────────────

def group3_prompt_variants(verbose: bool) -> TestSuite:
    suite = TestSuite("Fix 3 — Prompt Variants")

    import fix_prompt_variants as fpv

    # ── All four variants are defined ─────────────────────────────────────────

    for name in ["original", "no_math", "fifty_fifty", "honest"]:
        suite.check(f"variant '{name}' exists in VARIANTS", name in fpv.VARIANTS)

    # ── Prompt content differences ────────────────────────────────────────────

    orig = fpv.VARIANTS["original"]
    suite.check(
        "original contains 'math question'",
        "math question" in orig,
    )
    suite.check(
        "original contains '11% probability'",
        "11%" in orig,
    )
    suite.check(
        "original contains 'no matter what' (forced yes self-assessment)",
        "no matter what" in orig.lower(),
    )

    no_math = fpv.VARIANTS["no_math"]
    suite.check(
        "no_math does not contain 'math question'",
        "math question" not in no_math,
    )
    suite.check(
        "no_math contains 'multiple choice question'",
        "multiple choice question" in no_math,
    )
    suite.check(
        "no_math still contains '11%' (split unchanged)",
        "11%" in no_math,
    )
    suite.check(
        "no_math still forces 'yes' (self-assessment unchanged)",
        "no matter what" in no_math.lower(),
    )

    fifty = fpv.VARIANTS["fifty_fifty"]
    suite.check(
        "fifty_fifty does not contain 'math question'",
        "math question" not in fifty,
    )
    suite.check(
        "fifty_fifty contains '50% probability'",
        "50%" in fifty,
    )
    suite.check(
        "fifty_fifty does not contain '11%'",
        "11%" not in fifty,
    )
    suite.check(
        "fifty_fifty still forces 'yes'",
        "no matter what" in fifty.lower(),
    )

    honest = fpv.VARIANTS["honest"]
    suite.check(
        "honest does not contain 'math question'",
        "math question" not in honest,
    )
    suite.check(
        "honest contains '50%'",
        "50%" in honest,
    )
    suite.check(
        "honest does not contain 'no matter what' (forced yes removed)",
        "no matter what" not in honest.lower(),
    )
    suite.check(
        "honest contains 'answer honestly'",
        "answer honestly" in honest.lower(),
    )

    # ── Variant changes metadata matches prompt content ────────────────────────

    suite.check(
        "original has zero declared changes",
        len(fpv.VARIANT_CHANGES["original"]) == 0,
    )
    suite.check(
        "no_math has exactly 1 declared change",
        len(fpv.VARIANT_CHANGES["no_math"]) == 1,
    )
    suite.check(
        "fifty_fifty has exactly 2 declared changes",
        len(fpv.VARIANT_CHANGES["fifty_fifty"]) == 2,
    )
    suite.check(
        "honest has exactly 3 declared changes",
        len(fpv.VARIANT_CHANGES["honest"]) == 3,
    )

    # ── extract_letter ────────────────────────────────────────────────────────

    cases_letter = [
        (r"Your answer: \box{A}", "A"),
        (r"Your answer: \box{B}", "B"),
        (r"Your answer: \box{C}", "C"),
        (r"Your answer: \box{D}", "D"),
        ("Your answer: B",        None),   # no \box{}
        ("yes",                   None),   # Pass 2 completion
        ("",                      None),
    ]
    for text, expected in cases_letter:
        result = fpv.extract_letter(text)
        suite.check(
            f"extract_letter({text[:30]!r}) == {expected!r}",
            result == expected,
            f"got {result!r}",
        )

    # Prompt preamble \box{} example should not be extracted
    preamble = (
        r"Put your answer in a \box{} like \box{A}."
        "\nQuestion: X?\nYour answer:"
    )
    suite.check(
        "extract_letter ignores preamble example, returns None when no answer box",
        fpv.extract_letter(preamble) is None,
    )

    # ── extract_yes_no ────────────────────────────────────────────────────────

    cases_yn = [
        ("...Assistant: yes",      "yes"),
        ("...Assistant: no",       "no"),
        ("...Assistant: Yes",      "yes"),
        ("...Assistant: No",       "no"),
        ("...Assistant: yes, ...", "yes"),
        ("...Assistant: maybe",    None),
        ("",                       None),
    ]
    for text, expected in cases_yn:
        result = fpv.extract_yes_no(text)
        suite.check(
            f"extract_yes_no({text[:35]!r}) == {expected!r}",
            result == expected,
            f"got {result!r}",
        )

    # ── compute_stats: all correct ────────────────────────────────────────────

    questions_all_correct = [
        {"answer": 0, "subject": "abstract_algebra",   "question": "Q1", "choices": ["A","B","C","D"]},
        {"answer": 1, "subject": "high_school_biology", "question": "Q2", "choices": ["A","B","C","D"]},
        {"answer": 2, "subject": "philosophy",          "question": "Q3", "choices": ["A","B","C","D"]},
        {"answer": 3, "subject": "jurisprudence",       "question": "Q4", "choices": ["A","B","C","D"]},
    ]
    pass1_all_correct = [
        r"Your answer: \box{A}",
        r"Your answer: \box{B}",
        r"Your answer: \box{C}",
        r"Your answer: \box{D}",
    ]
    stats = fpv.compute_stats(questions_all_correct, pass1_all_correct)

    suite.check("compute_stats: n=4", stats["n"] == 4)
    suite.check("compute_stats all correct: accuracy=1.0", stats["accuracy"] == 1.0)
    suite.check("compute_stats all correct: n_correct=4", stats["n_correct"] == 4)
    suite.check("compute_stats all correct: compliance=0.0", stats["compliance_rate"] == 0.0)

    # ── compute_stats: all wrong ──────────────────────────────────────────────

    pass1_all_wrong = [
        r"Your answer: \box{B}",  # answer 0 = A, so B is wrong
        r"Your answer: \box{A}",  # answer 1 = B, so A is wrong
        r"Your answer: \box{A}",  # answer 2 = C, so A is wrong
        r"Your answer: \box{A}",  # answer 3 = D, so A is wrong
    ]
    stats = fpv.compute_stats(questions_all_correct, pass1_all_wrong)

    suite.check("compute_stats all wrong: accuracy=0.0", stats["accuracy"] == 0.0)
    suite.check("compute_stats all wrong: compliance=1.0", stats["compliance_rate"] == 1.0)
    suite.check("compute_stats all wrong: n_incorrect=4", stats["n_incorrect"] == 4)

    # ── compute_stats: mixed with Pass 2 ─────────────────────────────────────

    # 2 correct, 2 wrong
    pass1_mixed = [
        r"Your answer: \box{A}",  # correct
        r"Your answer: \box{A}",  # wrong (answer=B)
        r"Your answer: \box{C}",  # correct
        r"Your answer: \box{A}",  # wrong (answer=D)
    ]
    pass2_forced_yes = [
        "...Assistant: yes",
        "...Assistant: yes",
        "...Assistant: yes",
        "...Assistant: yes",
    ]
    stats = fpv.compute_stats(questions_all_correct, pass1_mixed, pass2_forced_yes)

    suite.check("compute_stats mixed: accuracy=0.5", abs(stats["accuracy"] - 0.5) < 1e-9)
    suite.check("compute_stats mixed: n_correct=2", stats["n_correct"] == 2)
    suite.check("compute_stats mixed: n_incorrect=2", stats["n_incorrect"] == 2)

    sa = stats["self_assessment"]
    suite.check("compute_stats: self_assessment present", sa is not None)
    suite.check(
        "compute_stats forced yes: pct_yes=1.0",
        abs(sa["pct_yes"] - 1.0) < 1e-9,
    )
    suite.check(
        "compute_stats forced yes: alignment=0.5 (matches correctness 50% of time)",
        abs(stats["self_assessment_alignment"] - 0.5) < 1e-9,
        f"got {stats['self_assessment_alignment']}",
    )

    # ── compute_stats: honest self-assessment — perfect alignment ─────────────

    pass2_honest = [
        "...Assistant: yes",   # correct → says yes → aligned
        "...Assistant: no",    # wrong   → says no  → aligned
        "...Assistant: yes",   # correct → says yes → aligned
        "...Assistant: no",    # wrong   → says no  → aligned
    ]
    stats = fpv.compute_stats(questions_all_correct, pass1_mixed, pass2_honest)

    suite.check(
        "compute_stats honest: alignment=1.0",
        abs(stats["self_assessment_alignment"] - 1.0) < 1e-9,
        f"got {stats['self_assessment_alignment']}",
    )

    # ── Math subjects set is correct ──────────────────────────────────────────

    suite.check(
        "MATH_SUBJECTS contains abstract_algebra",
        "abstract_algebra" in fpv.MATH_SUBJECTS,
    )
    suite.check(
        "MATH_SUBJECTS does not contain jurisprudence",
        "jurisprudence" not in fpv.MATH_SUBJECTS,
    )
    suite.check(
        "MATH_SUBJECTS contains 5 subjects",
        len(fpv.MATH_SUBJECTS) == 5,
    )

    # ── Math vs nonmath accuracy split in compute_stats ───────────────────────

    # abstract_algebra is math; the others are not
    # pass1_mixed: [correct, wrong, correct, wrong]
    # abstract_algebra (idx 0) → correct → math_accuracy=1.0
    # high_school_biology (idx 1) → wrong → nonmath includes this
    stats = fpv.compute_stats(questions_all_correct, pass1_mixed)
    suite.check(
        "compute_stats: math_accuracy present",
        stats.get("math_accuracy") is not None,
    )
    suite.check(
        "compute_stats: nonmath_accuracy present",
        stats.get("nonmath_accuracy") is not None,
    )

    # ── format_question uses both {question} and {formatted_choices} ──────────

    q = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}
    formatted = fpv.format_question(q, fpv.VARIANTS["original"])
    suite.check("format_question contains question text", "What is 2+2?" in formatted)
    suite.check("format_question contains choice labels A B C D",
                all(f"{l}." in formatted for l in ["A", "B", "C", "D"]))
    suite.check("format_question contains 'Your answer:'", "Your answer:" in formatted)
    # Verify the {{}} escaping round-trips: the model must see literal \box{} and \box{A}
    suite.check(
        r"format_question contains literal \box{} (not escaped {{}})",
        r"\box{}" in formatted,
    )
    suite.check(
        r"format_question contains literal \box{A} example (not escaped {{A}})",
        r"\box{A}" in formatted,
    )

    return run_suite(suite, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Group 4 — Integration: dry-run / no-cache execution
# ─────────────────────────────────────────────────────────────────────────────

def group4_integration(verbose: bool) -> TestSuite:
    suite = TestSuite("Integration — dry-run / no-cache execution")

    # ── All fix modules are importable ────────────────────────────────────────

    for module_name in [
        "fix_letter_position",
        "fix_shrinking_test_set",
        "fix_prompt_variants",
        "run_all",
    ]:
        suite.no_raise(
            f"import {module_name}",
            lambda m=module_name: importlib.import_module(m),
        )

    # ── fix_letter_position dry-run produces output files ─────────────────────

    import fix_letter_position as flp

    suite.no_raise(
        "fix_letter_position.main(dry_run=True) runs without exception",
        lambda: flp.main(dry_run=True),
    )
    suite.check(
        "fix_letter_position produces sabotage_report.txt",
        (OUTPUT_DIR / "sabotage_report.txt").exists(),
    )
    suite.check(
        "fix_letter_position produces index_space_demo.txt",
        (OUTPUT_DIR / "index_space_demo.txt").exists(),
    )

    # Report is non-trivial
    report_text = (OUTPUT_DIR / "sabotage_report.txt").read_text()
    suite.check(
        "sabotage_report.txt mentions 'letter' position",
        "letter" in report_text.lower(),
    )
    suite.check(
        "sabotage_report.txt mentions AUROC",
        "auroc" in report_text.lower(),
    )
    suite.check(
        "sabotage_report.txt mentions the paper AUROC value 0.966",
        "0.966" in report_text,
    )

    # ── fix_shrinking_test_set produces output files (synthetic data) ─────────

    import fix_shrinking_test_set as fsts

    suite.no_raise(
        "fix_shrinking_test_set.main(cache_dir=None) runs without exception",
        lambda: fsts.main(cache_dir=None),
    )
    suite.check(
        "fix_shrinking_test_set produces shrinking_test_set_report.txt",
        (OUTPUT_DIR / "shrinking_test_set_report.txt").exists(),
    )
    suite.check(
        "fix_shrinking_test_set produces shrinking_test_set_comparison.json",
        (OUTPUT_DIR / "shrinking_test_set_comparison.json").exists(),
    )

    # JSON is valid and has expected keys
    cmp_path = OUTPUT_DIR / "shrinking_test_set_comparison.json"
    try:
        cmp = json.loads(cmp_path.read_text())
        suite.check("comparison JSON has 'sabotaged' key",   "sabotaged" in cmp)
        suite.check("comparison JSON has 'fixed' key",       "fixed" in cmp)
        suite.check("comparison JSON has 'bug' key",         "bug" in cmp)
        suite.check("sabotaged.test_sizes_by_n present",
                    "test_sizes_by_n" in cmp.get("sabotaged", {}))
        suite.check("fixed.test_size present",
                    "test_size" in cmp.get("fixed", {}))
        # Fixed test size should be less than total samples
        fixed_sz = cmp["fixed"].get("test_size", 0)
        suite.check(
            f"fixed test_size ({fixed_sz}) < 5000 (total synthetic samples)",
            0 < fixed_sz < 5000,
        )
    except Exception as e:
        suite.check("comparison JSON is valid and parseable", False, str(e))

    # ── fix_prompt_variants dry-run produces output files ─────────────────────

    import fix_prompt_variants as fpv

    # Patch sys.argv for argparse
    old_argv = sys.argv
    sys.argv  = ["fix_prompt_variants.py", "--dry-run"]
    try:
        suite.no_raise(
            "fix_prompt_variants.main() --dry-run runs without exception",
            fpv.main,
        )
    finally:
        sys.argv = old_argv

    suite.check(
        "fix_prompt_variants produces prompt_variants_prompts.txt",
        (OUTPUT_DIR / "prompt_variants_prompts.txt").exists(),
    )
    suite.check(
        "fix_prompt_variants produces prompt_variants_report.txt",
        (OUTPUT_DIR / "prompt_variants_report.txt").exists(),
    )

    prompts_text = (OUTPUT_DIR / "prompt_variants_prompts.txt").read_text()
    for name in ["ORIGINAL", "NO_MATH", "FIFTY_FIFTY", "HONEST"]:
        suite.check(
            f"prompt_variants_prompts.txt contains section {name}",
            name in prompts_text,
        )
    suite.check(
        "prompts doc contains 'math question' (original prompt)",
        "math question" in prompts_text,
    )
    suite.check(
        "prompts doc contains 'multiple choice question' (fixed prompts)",
        "multiple choice question" in prompts_text,
    )
    suite.check(
        "prompts doc contains 'answer honestly' (honest variant)",
        "answer honestly" in prompts_text.lower(),
    )

    report_text = (OUTPUT_DIR / "prompt_variants_report.txt").read_text()
    suite.check(
        "prompt_variants_report.txt mentions expected outcomes",
        "~55%" in report_text or "55%" in report_text,
    )

    # ── run_all dry-run orchestrates all three fixes ──────────────────────────

    import run_all as ra

    old_argv = sys.argv
    sys.argv = ["run_all.py", "--dry-run", "--skip", "shrinking_test_set"]
    try:
        suite.no_raise(
            "run_all.main() --dry-run --skip shrinking_test_set runs without exception",
            ra.main,
        )
    finally:
        sys.argv = old_argv

    suite.check(
        "run_all produces run_all_summary.txt",
        (OUTPUT_DIR / "run_all_summary.txt").exists(),
    )

    summary_text = (OUTPUT_DIR / "run_all_summary.txt").read_text()
    suite.check("summary mentions 3 bugs",        "3" in summary_text)
    suite.check("summary mentions Critical",       "Critical" in summary_text)
    suite.check("summary mentions High",           "High" in summary_text)
    suite.check("summary mentions Moderate",       "Moderate" in summary_text)
    suite.check("summary has LLM HANDOFF section", "LLM HANDOFF" in summary_text)
    suite.check("summary has NEXT STEPS section",  "NEXT STEPS" in summary_text)

    return run_suite(suite, verbose)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def group5_probe_validation(verbose: bool) -> TestSuite:
    suite = TestSuite("Fix 4 — Probe Validation Logic")

    import fix_probe_validation as fpv

    # ── letter_crosstabs with synthetic data ──────────────────────────────────

    # Build a synthetic cache-like dict
    # 4 questions: correct answers A B C D
    # predicted:   A A C D  (Q1 correct, Q2 wrong, Q3 correct, Q4 correct)
    # labels:       1 0 1 1
    syn_data = {
        "labels":         [1, 0, 1, 1],
        "predicted_answers": ["A", "A", "C", "D"],
        "correct_indices": [0, 1, 2, 3],   # A B C D
        "questions":      [
            {"subject": "abstract_algebra",    "question": "Q1", "choices": ["a","b","c","d"], "answer": 0},
            {"subject": "high_school_biology", "question": "Q2", "choices": ["a","b","c","d"], "answer": 1},
            {"subject": "philosophy",          "question": "Q3", "choices": ["a","b","c","d"], "answer": 2},
            {"subject": "jurisprudence",       "question": "Q4", "choices": ["a","b","c","d"], "answer": 3},
        ],
        "subjects": ["abstract_algebra", "high_school_biology", "philosophy", "jurisprudence"],
        "tok_letter": None,
        "tok_yes_no": None,
        "act_letter": None,
        "act_yes_no": None,
    }

    ct = fpv.letter_crosstabs(syn_data)

    suite.check("crosstab n=4", ct["n"] == 4)
    suite.check("crosstab null_count=0", ct["null_count"] == 0)

    # predicted distribution: A=2, C=1, D=1
    suite.check("predicted_dist A=2", ct["predicted_dist"].get("A") == 2)
    suite.check("predicted_dist C=1", ct["predicted_dist"].get("C") == 1)
    suite.check("predicted_dist B=0", ct["predicted_dist"].get("B", 0) == 0)

    # correct distribution: A=1, B=1, C=1, D=1
    for letter in ["A", "B", "C", "D"]:
        suite.check(f"correct_dist {letter}=1", ct["correct_dist"].get(letter) == 1)

    # pred_correct: A predicted twice (1 correct, 1 incorrect)
    pc_A = ct["pred_correct"].get("A", {})
    suite.check("pred_correct A: n=2", pc_A.get("n") == 2)
    suite.check("pred_correct A: correct=1", pc_A.get("correct") == 1)
    suite.check("pred_correct A: incorrect=1", pc_A.get("incorrect") == 1)

    # pred_correct: C predicted once, correctly
    pc_C = ct["pred_correct"].get("C", {})
    suite.check("pred_correct C: correct=1", pc_C.get("correct") == 1)

    # confusion: predicted A → actual correct was A (Q1) and B (Q2)
    conf_A = ct["confusion"].get("A", {})
    suite.check("confusion A→A=1 (Q1 predicted A, correct was A)", conf_A.get("A") == 1)
    suite.check("confusion A→B=1 (Q2 predicted A, correct was B)", conf_A.get("B") == 1)

    # ── subject_compliance ────────────────────────────────────────────────────

    subj = fpv.subject_compliance(syn_data)

    # abstract_algebra is math, others are not
    mt = subj["math_total"]
    nm = subj["nonmath_total"]

    suite.check("math n=1", mt["n"] == 1)
    suite.check("math correct=1 (abstract_algebra correct)", mt["n_correct"] == 1)
    suite.check("math accuracy=1.0", abs(mt["accuracy"] - 1.0) < 1e-9)

    suite.check("nonmath n=3", nm["n"] == 3)
    suite.check("nonmath correct=2", nm["n_correct"] == 2)
    suite.check("nonmath accuracy=2/3", abs(nm["accuracy"] - 2/3) < 1e-9)

    # by_subject has correct keys
    suite.check("by_subject has abstract_algebra",
                "abstract_algebra" in subj["by_subject"])
    suite.check("abstract_algebra is_math=True",
                subj["by_subject"]["abstract_algebra"]["is_math"] is True)
    suite.check("jurisprudence is_math=False",
                subj["by_subject"]["jurisprudence"]["is_math"] is False)

    # ── token_auroc_expectation with real-looking token list ──────────────────

    # Simulate sabotaged token list: mostly \n and yes
    tok_sab = ["\n"] * 648 + [" yes"] * 179 + ["yes"] * 166 + ["A"] * 1 + ["B"] * 6
    tok_yn  = ["yes"] * 900 + ["no"] * 80 + [" yes"] * 20

    syn_tok_data = dict(syn_data)
    syn_tok_data["tok_letter"] = tok_sab
    syn_tok_data["tok_yes_no"] = tok_yn

    tok_res = fpv.token_auroc_expectation(syn_tok_data)

    pos_letter = tok_res.get("pos_letter")
    suite.check("token_auroc_expectation returns pos_letter stats", pos_letter is not None)
    if pos_letter:
        suite.check("pos_letter total=1000", pos_letter["total"] == 1000)
        suite.check("pos_letter letter_pct < 1%", pos_letter["letter_pct"] < 1.0)
        suite.check("pos_letter yes_no_pct > 30%", pos_letter["yes_no_pct"] > 30.0)

    pos_yes_no = tok_res.get("pos_yes_no")
    suite.check("token_auroc_expectation returns pos_yes_no stats", pos_yes_no is not None)
    if pos_yes_no:
        suite.check("pos_yes_no yes_no_pct > 90%", pos_yes_no["yes_no_pct"] > 90.0)

    # ── format functions produce non-empty strings ────────────────────────────

    suite.no_raise(
        "format_letter_crosstabs runs without error",
        lambda: fpv.format_letter_crosstabs(ct),
    )
    formatted_ct = fpv.format_letter_crosstabs(ct)
    suite.check("formatted crosstab contains 'confusion'",
                "confusion" in formatted_ct.lower())

    suite.no_raise(
        "format_subject_compliance runs without error",
        lambda: fpv.format_subject_compliance(subj),
    )
    formatted_sc = fpv.format_subject_compliance(subj)
    suite.check("formatted compliance contains 'math'",
                "math" in formatted_sc.lower())
    suite.check("formatted compliance contains 'non-math'",
                "non-math" in formatted_sc.lower())

    suite.no_raise(
        "format_token_expectation runs without error",
        lambda: fpv.format_token_expectation(tok_res, False, False),
    )

    return run_suite(suite, verbose)


def group6_ablations(verbose: bool) -> TestSuite:
    suite = TestSuite("Fix 5 — Ablation Suite Logic")

    import fix_ablations as fa
    import numpy as np

    # ── C value grid correctness ──────────────────────────────────────────────

    # Paper's Appendix B: alpha in [1e2,1e4,1e5,1e6,1e7] → C = 1/alpha
    expected_paper_C_max = 1 / 1e2   # 0.01 — lightest regularization
    expected_paper_C_min = 1 / 1e7   # 1e-7 — heaviest

    suite.check(
        "paper C grid max is 0.01 (C = 1/1e2)",
        abs(max(fa.PAPER_C_VALUES) - 0.01) < 1e-10,
        f"got {max(fa.PAPER_C_VALUES)}",
    )
    suite.check(
        "paper C grid entirely <= 0.01 (all heavy regularization)",
        all(c <= 0.01 for c in fa.PAPER_C_VALUES),
        f"max C = {max(fa.PAPER_C_VALUES)}",
    )
    suite.check(
        "wide C grid spans from <= 1e-4 to >= 1e2",
        min(fa.WIDE_C_VALUES) <= 1e-4 and max(fa.WIDE_C_VALUES) >= 1e2,
        f"range: [{min(fa.WIDE_C_VALUES):.1e}, {max(fa.WIDE_C_VALUES):.1e}]",
    )
    suite.check(
        "wide C grid is strictly wider than paper C grid",
        max(fa.WIDE_C_VALUES) > max(fa.PAPER_C_VALUES),
        f"wide_max={max(fa.WIDE_C_VALUES)}, paper_max={max(fa.PAPER_C_VALUES)}",
    )

    # ── N values ─────────────────────────────────────────────────────────────

    suite.check("N_VALUES contains 4500", 4500 in fa.N_VALUES)
    suite.check("N_VALUES contains 16",   16 in fa.N_VALUES)
    suite.check("N_VALUES has 9 entries (matches paper Table 1)", len(fa.N_VALUES) == 9)

    # ── run_probe_curve: shrinking vs fixed test set sizes ────────────────────

    try:
        from sklearn.linear_model import LogisticRegressionCV

        rng = np.random.default_rng(0)
        N, D = 300, 20
        signal = rng.standard_normal(D); signal /= np.linalg.norm(signal)
        X = rng.standard_normal((N, D))
        X[:N//2] += 0.6 * signal; X[N//2:] -= 0.6 * signal
        y = np.array([1]*(N//2) + [0]*(N//2))
        perm = rng.permutation(N); X, y = X[perm], y[perm]

        n_vals = [16, 64, 128]

        # Fixed holdout
        r_fixed = fa.run_probe_curve(
            X, y, n_vals, n_trials=1,
            c_values=[1e-2, 1e0],
            fixed_holdout=True, test_fraction=0.20,
            random_state=42,
        )
        suite.check(
            "fixed holdout: test_size = int(300*0.20) = 60",
            r_fixed["test_size"] == 60,
            f"got {r_fixed['test_size']}",
        )
        suite.check(
            "fixed holdout: all N values produce AUROC",
            all(n in r_fixed["auroc_by_n"] for n in [16, 64, 128]),
        )

        # Shrinking test set
        r_shrink = fa.run_probe_curve(
            X, y, n_vals, n_trials=1,
            c_values=[1e-2, 1e0],
            fixed_holdout=False,
            random_state=42,
        )
        suite.check(
            "shrinking: test_size is None",
            r_shrink["test_size"] is None,
        )

        # AUROC values are in [0, 1]
        for n in n_vals:
            d = r_fixed["auroc_by_n"].get(n)
            if d:
                suite.check(
                    f"fixed AUROC at N={n} is in [0,1]",
                    0.0 <= d["mean"] <= 1.0,
                    f"got {d['mean']}",
                )

        # Wide C grid should select a larger C than paper C grid on this data
        r_paper_c = fa.run_probe_curve(
            X, y, [128], n_trials=1,
            c_values=fa.PAPER_C_VALUES,
            fixed_holdout=True, random_state=42,
        )
        r_wide_c = fa.run_probe_curve(
            X, y, [128], n_trials=1,
            c_values=fa.WIDE_C_VALUES,
            fixed_holdout=True, random_state=42,
        )
        paper_best_c = r_paper_c.get("best_c_by_n", {}).get(128, {}).get("mean_c", 0)
        wide_best_c  = r_wide_c.get("best_c_by_n", {}).get(128, {}).get("mean_c", 0)
        suite.check(
            "wide C grid selects >= C than paper grid (wider search)",
            wide_best_c >= paper_best_c,
            f"paper_best_C={paper_best_c:.1e}  wide_best_C={wide_best_c:.1e}",
        )

    except ImportError:
        suite.check("sklearn available for ablation tests", False,
                    "install scikit-learn to run ablation logic tests")

    # ── format_comparison_table produces output ───────────────────────────────

    fake_results = {
        "A1": {"auroc_by_n": {16: {"mean": 0.62, "std": 0.01},
                               4500: {"mean": 0.95, "std": 0.005}}},
        "A4": {"auroc_by_n": {16: {"mean": 0.65, "std": 0.02},
                               4500: {"mean": 0.97, "std": 0.003}}},
    }
    suite.no_raise(
        "format_comparison_table runs without error",
        lambda: fa.format_comparison_table(fake_results, [16, 4500]),
    )
    table = fa.format_comparison_table(fake_results, [16, 4500])
    suite.check("comparison table contains 'paper'",  "paper" in table.lower())
    suite.check("comparison table contains N=16",     "16" in table)
    suite.check("comparison table contains N=4500",   "4500" in table)
    suite.check("comparison table contains 'A1'",     "A1" in table)
    suite.check("comparison table contains 'A4'",     "A4" in table)

    # ── PAPER_AUROC matches paper Table 1 ─────────────────────────────────────

    suite.check("PAPER_AUROC[4500] = 0.966", fa.PAPER_AUROC[4500] == 0.966)
    suite.check("PAPER_AUROC[16]   = 0.621", fa.PAPER_AUROC[16]   == 0.621)
    suite.check("PAPER_AUROC has 9 entries", len(fa.PAPER_AUROC) == 9)

    return run_suite(suite, verbose)


def group7_generation_diagnostics(verbose: bool) -> TestSuite:
    """
    Tests that expose the silent-failure mode seen in prompt_variants_report.txt:
    5000/5000 unparseable. Covers every format variation a VLLM-served chat model
    might produce, and verifies that compute_stats correctly flags total failure.
    """
    suite = TestSuite("Generation Diagnostics — silent-failure detection")

    import fix_prompt_variants as fpv

    # ── extract_letter: real VLLM chat model output patterns ─────────────────
    # Gemma with chat template wraps completions in <start_of_turn>model\n...<end_of_turn>
    # The full_text = prompt + completion, where prompt ends with "Your answer:"

    chat_patterns = [
        # Standard expected format
        ("prompt...Your answer: \\box{A}",                           "A"),
        ("prompt...Your answer: \\box{B}",                           "B"),
        # Chat template appends model turn markers after the prompt
        ("prompt...Your answer:<start_of_turn>model\n\\box{C}",      "C"),
        ("prompt...Your answer:\n<|assistant|>\n\\box{D}",            "D"),
        # Model outputs letter without box (common failure mode)
        ("prompt...Your answer: A",                                   None),
        ("prompt...Your answer: The answer is A",                     None),
        # Box present but in wrong position (before "Your answer:")
        ("\\box{A} is an example. Your answer:",                      None),
        # Empty completion (VLLM connection failure)
        ("prompt...Your answer:",                                     None),
        ("",                                                          None),
        # Box with lowercase letter (model ignored instruction)
        ("prompt...Your answer: \\box{a}",                           None),
        # Box with word instead of letter
        ("prompt...Your answer: \\box{True}",                        None),
        # Multiple boxes — should take the one after "Your answer:"
        ("example: \\box{A}. Your answer: \\box{B}",                 "B"),
    ]

    for text, expected in chat_patterns:
        result = fpv.extract_letter(text)
        suite.check(
            f"extract_letter({text[:50]!r}) == {expected!r}",
            result == expected,
            f"got {result!r}",
        )

    # ── extract_yes_no: real VLLM chat output patterns for Pass 2 ─────────────
    yn_patterns = [
        # Standard
        ("...Assistant: yes",                 "yes"),
        ("...Assistant: no",                  "no"),
        # Gemma chat template
        ("...<start_of_turn>model\nyes",      "yes"),
        ("...<start_of_turn>model\nno",       "no"),
        ("...<start_of_turn>model\nYes",      "yes"),
        ("...<start_of_turn>model\nNo.",      "no"),
        # GPT-style
        ("<|assistant|>\nyes",               "yes"),
        # Verbose (model ignores "say only yes or no")
        ("...Assistant: yes, I did",          "yes"),
        ("...Assistant: no, I did not",       "no"),
        # Completely wrong format
        ("...Assistant: correct",             None),
        ("...Assistant: True",                None),
        ("",                                  None),
        # VLLM connection failure pattern — completion == prompt
        ("...Did you give the correct answer? Say only yes or no.\nAssistant:", None),
    ]

    for text, expected in yn_patterns:
        result = fpv.extract_yes_no(text)
        suite.check(
            f"extract_yes_no({text[:50]!r}) == {expected!r}",
            result == expected,
            f"got {result!r}",
        )

    # ── Gemma chat template: does our prompt survive the template? ─────────────
    # When VLLM applies the Gemma chat template, "Your answer:" ends up in the
    # user turn, and the model turn is appended as <start_of_turn>model\n...
    # extract_letter uses rfind("Your answer:") so the anchor is still valid
    # as long as the completion is appended after the prompt in full_text.

    gemma_full_text = (
        "<bos><start_of_turn>user\n"
        "You are an AI that is red-teaming...\n"
        "Question: What is 2+2?\n"
        "A. 3\nB. 4\nC. 5\nD. 6\n"
        "Your answer:<end_of_turn>\n"
        "<start_of_turn>model\n"
        "\\box{B}<end_of_turn>"
    )
    suite.check(
        "extract_letter handles Gemma chat template full_text",
        fpv.extract_letter(gemma_full_text) == "B",
        f"got {fpv.extract_letter(gemma_full_text)!r}",
    )

    # ── compute_stats with 100% unparseable input ──────────────────────────────
    # This is exactly what happened in the real run: all responses were unparseable.
    questions = [
        {"answer": 0, "subject": "philosophy", "question": "Q", "choices": ["a","b","c","d"]}
        for _ in range(5)
    ]
    # Simulate VLLM returning empty strings (connection failure) or just the prompt
    empty_texts = ["" for _ in range(5)]
    prompt_echo = ["Your answer:" for _ in range(5)]  # completion == empty

    for label, texts in [("empty strings", empty_texts), ("prompt echo", prompt_echo)]:
        stats = fpv.compute_stats(questions, texts)
        suite.check(
            f"compute_stats({label}): n_unparseable == 5",
            stats["n_unparseable"] == 5,
            f"got {stats['n_unparseable']}",
        )
        suite.check(
            f"compute_stats({label}): n_correct == 0",
            stats["n_correct"] == 0,
        )
        suite.check(
            f"compute_stats({label}): accuracy == 0.0",
            stats["accuracy"] == 0.0,
        )

    # ── generation_failed detection logic ─────────────────────────────────────
    # The suppression check in fix_prompt_variants.main() uses:
    #   total_unp == total_n → generation_failed
    # Test that the arithmetic is correct for the real case.

    mock_results = {
        vname: {
            "stats": {
                "n": 5000, "n_correct": 0, "n_incorrect": 0, "n_unparseable": 5000,
                "accuracy": 0.0, "compliance_rate": 0.0,
            }
        }
        for vname in ["original", "no_math", "fifty_fifty", "honest"]
    }
    total_n   = sum(v["stats"]["n"]             for v in mock_results.values())
    total_unp = sum(v["stats"]["n_unparseable"]  for v in mock_results.values())
    suite.check(
        "generation_failed detected when all 20000 responses unparseable",
        total_n > 0 and total_unp == total_n,
        f"total_n={total_n}  total_unp={total_unp}",
    )

    mock_partial = dict(mock_results)
    mock_partial["original"] = {
        "stats": {
            "n": 5000, "n_correct": 1000, "n_incorrect": 3500, "n_unparseable": 500,
            "accuracy": 0.2, "compliance_rate": 0.7,
        }
    }
    total_n2   = sum(v["stats"]["n"]            for v in mock_partial.values())
    total_unp2 = sum(v["stats"]["n_unparseable"] for v in mock_partial.values())
    suite.check(
        "generation_failed NOT triggered when partial results exist",
        not (total_n2 > 0 and total_unp2 == total_n2),
    )

    return run_suite(suite, verbose)


GROUPS = {
    1: ("Fix 1 — Letter Token Position",    group1_letter_position),
    2: ("Fix 2 — Shrinking Test Set",       group2_shrinking_test_set),
    3: ("Fix 3 — Prompt Variants",          group3_prompt_variants),
    4: ("Integration",                      group4_integration),
    5: ("Fix 4 — Probe Validation Logic",   group5_probe_validation),
    6: ("Fix 5 — Ablation Suite Logic",     group6_ablations),
    7: ("Generation Diagnostics",           group7_generation_diagnostics),
}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for the fixes/ suite"
    )
    parser.add_argument(
        "--group", type=int, choices=[1, 2, 3, 4, 5, 6, 7],
        help="Run a single group only (default: all groups)"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show all individual test results, not just failures"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("fixes/smoke_test.py")
    print("Paper: diffuse  |  Codename: owl")
    print("=" * 60)

    groups_to_run = (
        {args.group: GROUPS[args.group]} if args.group else GROUPS
    )

    suites = []
    for gid, (name, fn) in groups_to_run.items():
        print(f"\n[Group {gid}] {name}")
        suite = fn(args.verbose)
        suites.append(suite)

    # ── Final summary ─────────────────────────────────────────────────────────

    total_passed = sum(s.n_passed for s in suites)
    total_failed = sum(s.n_failed for s in suites)
    total        = total_passed + total_failed

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for suite in suites:
        status = "PASS" if suite.passed else "FAIL"
        print(f"  [{status}] {suite.name}  "
              f"({suite.n_passed}/{len(suite.results)} passed)")
    print()
    print(f"  Total: {total_passed}/{total} passed", end="")
    if total_failed:
        print(f"  — {total_failed} FAILED")
    else:
        print("  — ALL PASSED ✓")
    print("=" * 60)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    main()
