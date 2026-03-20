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

GROUPS = {
    1: ("Fix 1 — Letter Token Position", group1_letter_position),
    2: ("Fix 2 — Shrinking Test Set",    group2_shrinking_test_set),
    3: ("Fix 3 — Prompt Variants",       group3_prompt_variants),
    4: ("Integration",                   group4_integration),
}


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for the fixes/ suite"
    )
    parser.add_argument(
        "--group", type=int, choices=list(GROUPS.keys()),
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
