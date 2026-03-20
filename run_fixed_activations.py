#!/usr/bin/env python3
"""
run_fixed_activations.py

Extracts activations at the CORRECTED letter token position and runs ablations.

Speedups vs naive version:
  - Patches src/activations.py in the container before importing, so the
    existing multi-GPU extraction function is used as-is.
  - Splits 5000 texts across NUM_GPUS A100s in parallel with batched
    forward passes — no manual text-by-text Python loop.

Usage:
    modal run run_fixed_activations.py

Outputs:
    fixed_cache/          corrected .npy activation files
    fixes/output/fixed_ablations_report.txt
    fixes/output/sabotaged_ablations_report.txt
"""

import json
from pathlib import Path

import modal

app = modal.App("diffuse-fixed-activations")

NUM_GPUS = 2        # split 5000 texts across this many A100-80GBs
LAYER = 30
POSITIONS = ["letter", "letter+1", "letter-1", "yes_no", "yes_no-1", "last"]

diffuse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch", "transformers", "scikit-learn", "matplotlib",
        "numpy", "scipy", "accelerate", "hf_transfer",
    )
    .add_local_dir(
        str(Path(__file__).parent),
        remote_path="/workspace",
        copy=True,
        ignore=["__pycache__", "*.egg-info", ".venv", "logs"],
    )
)

hf_cache = modal.Volume.from_name("hf-model-cache", create_if_missing=True)


@app.function(
    image=diffuse_image,
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=3 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def extract_fixed_activations() -> dict:
    import os, sys, json, shutil
    import numpy as np
    from pathlib import Path

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/scripts")
    sys.path.insert(0, "/workspace/fixes")

    hf_token = os.environ.get("HF_TOKEN") or ""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        p = Path("/root/.cache/huggingface/token")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(hf_token)

    # ── Patch activations.py before any import ────────────────────────────────
    # String-patch the file so the existing multi-GPU function uses the fix.
    act_path = Path("/workspace/src/activations.py")
    src = act_path.read_text()

    # Build the fixed function as a string to inject into activations.py.
    # We insert it right before find_yes_no_token_position (a stable anchor)
    # so it ends up at module scope. The triple-backslash for the box pattern
    # becomes a single backslash in the written file, which is what the
    # tokenizer will see at runtime.
    # Fixed function: scans for ALL \box{[A-D]} occurrences, returns the LAST.
    # Why last: the prompt instructions contain \box{A} as an example (appears
    # first in the text). The model's actual answer is always the last occurrence.
    # Anchoring to "answer" is wrong for Pass-2 full texts: "correct answer"
    # in the Pass-2 question sets search_start AFTER \box{X}, finding nothing.
    fixed_fn_lines = [
        "",
        "",
        "def find_letter_token_position_in_full_text(full_text, tokenizer):",
        '    """Fixed: finds last \\box{[A-D]} in full text, returns full-text index."""',
        "    full_tokens = tokenizer.tokenize(full_text)",
        "    last_match = None",
        "    for i in range(len(full_tokens) - 4):",
        "        if chr(92) in full_tokens[i] and full_tokens[i+1:i+3] == ['box', '{']:",
        "            candidate = full_tokens[i+3]",
        "            if candidate.strip() in ['A', 'B', 'C', 'D']:",
        "                last_match = (i + 3, candidate)",
        "    if last_match:",
        "        return last_match",
        "    return None, None",
        "",
    ]
    fixed_fn = "\n".join(fixed_fn_lines) + "\n"

    if "find_letter_token_position_in_full_text" not in src:
        marker = "\ndef find_yes_no_token_position("
        if marker in src:
            idx = src.index(marker)
            src = src[:idx] + fixed_fn + src[idx:]
            print("[patch] fixed function inserted before find_yes_no_token_position")
        else:
            src = fixed_fn + src
            print("[patch] fixed function prepended (anchor not found)")
    else:
        print("[patch] fixed function already present")

    # Replace the call site to use full_text instead of completion
    old_call = (
        '                        if position in ["letter", "letter+1", "letter-1"]:\n'
        '                            completion = completions[text_idx] if completions else ""\n'
        '                            letter_pos, probed_token_str = find_letter_token_position(\n'
        '                                completion, tokenizer\n'
        '                            )\n'
    )
    new_call = (
        '                        if position in ["letter", "letter+1", "letter-1"]:\n'
        '                            letter_pos, probed_token_str = find_letter_token_position_in_full_text(\n'
        '                                full_text, tokenizer\n'
        '                            )\n'
    )
    if old_call in src:
        src = src.replace(old_call, new_call)
        print("[patch] call site replaced — using fixed letter-position finder")
    else:
        print("[patch] WARNING: call site not found — check activations.py has not changed")

    act_path.write_text(src)
    print("[patch] src/activations.py patched")

    # ── Import after patch ────────────────────────────────────────────────────
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    from config import config
    from src.activations import extract_all_layers_all_positions_multi_gpu

    # ── Load generation cache (no VLLM needed) ────────────────────────────────
    gen_dir = Path("experiments/gemma-3-12b/generations")
    p2_base = gen_dir / "lie_detector_part2/all/n5000_t1.0_maxtok100"

    full_texts_p2   = np.load(str(p2_base) + "_full_texts.npy",   allow_pickle=True).tolist()
    completions_p2  = np.load(str(p2_base) + "_completions.npy",  allow_pickle=True).tolist()
    print(f"Loaded {len(full_texts_p2)} Pass-2 conversations from cache")

    # ── Multi-GPU extraction ───────────────────────────────────────────────────
    # completions_p2 is still passed for yes_no position logic (unaffected by
    # the patch). For letter positions the patch ignores it and uses full_text.
    gpu_ids = list(range(NUM_GPUS))
    print(f"\nExtracting layer={LAYER}, positions={POSITIONS}, gpus={gpu_ids}...")

    layer_pos_acts, probed_tokens = extract_all_layers_all_positions_multi_gpu(
        model_name=config.MODEL_NAME,
        full_texts=full_texts_p2,
        layer_indices=[LAYER],
        positions_to_extract=POSITIONS,
        num_gpus=NUM_GPUS,
        batch_size=config.ACTIVATION_BATCH_SIZE,
        use_model_cache=True,
        gpu_ids=gpu_ids,
        completions=completions_p2,
    )

    # ── Token quality check ───────────────────────────────────────────────────
    print("\nToken quality check:")
    for pos in POSITIONS:
        toks = probed_tokens[pos]
        n = max(len(toks), 1)
        letter_pct = 100 * sum(1 for t in toks if t and t.strip() in ["A","B","C","D"]) / n
        yes_pct    = 100 * sum(1 for t in toks if t and "yes" in t.lower()) / n
        nl_pct     = 100 * sum(1 for t in toks if t == "\n" or (t and not t.strip())) / n
        print(f"  {pos:14}  A/B/C/D={letter_pct:5.1f}%  yes={yes_pct:5.1f}%  newline={nl_pct:5.1f}%")

    # ── Save fixed cache ──────────────────────────────────────────────────────
    orig_dir = Path("experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered")
    out_dir  = Path("experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered_fixed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in POSITIONS:
        arr = layer_pos_acts[LAYER][pos]
        np.save(out_dir / f"activations_layer{LAYER:02d}_pos-{pos}.npy", arr)
        with open(out_dir / f"probed_tokens_pos-{pos}.json", "w") as f:
            json.dump(probed_tokens[pos], f, indent=2)

    for fname in [
        "labels.json", "labels.npy", "questions.json", "subjects.json",
        "predicted_answers.json", "correct_answer_indices.json",
        "prompts.json", "generated.json", "completions.json", "first_generation.json",
    ]:
        src_f = orig_dir / fname
        if src_f.exists():
            shutil.copy(src_f, out_dir / fname)

    print(f"\nFixed cache saved: {out_dir}")

    # ── Ablations on both caches ──────────────────────────────────────────────
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)

    for label, cache_path in [("sabotaged", str(orig_dir)), ("fixed", str(out_dir))]:
        print(f"\n{'='*60}\nAblations — {label.upper()}\n{'='*60}")
        _run_ablations(cache_path, label)

    # ── Collect outputs ───────────────────────────────────────────────────────
    results = {}
    for pos in POSITIONS:
        for suffix in [".npy", ".json"]:
            if suffix == ".npy":
                p = out_dir / f"activations_layer{LAYER:02d}_pos-{pos}.npy"
                k = f"fixed_cache/activations_layer{LAYER:02d}_pos-{pos}.npy"
            else:
                p = out_dir / f"probed_tokens_pos-{pos}.json"
                k = f"fixed_cache/probed_tokens_pos-{pos}.json"
            if p.exists():
                results[k] = p.read_bytes()

    for label in ["sabotaged", "fixed"]:
        for fname in ["ablations_report.txt", "ablations.json"]:
            p = Path(f"fixes/output/{label}_{fname}")
            if p.exists():
                results[f"fixes/output/{label}_{fname}"] = p.read_bytes()

    hf_cache.commit()
    return results


def _run_ablations(cache_dir: str, output_prefix: str):
    import sys, shutil
    from pathlib import Path
    sys.path.insert(0, "/workspace/fixes")
    import fix_ablations as fa
    old_argv = sys.argv
    sys.argv = [
        "fix_ablations.py",
        "--cache-dir", cache_dir,
        "--layer", str(LAYER),
        "--n-trials", "5",
        "--ablations", "A1", "A2", "A3", "A4", "A5", "A7",
    ]
    try:
        fa.main()
    except (SystemExit, Exception) as e:
        print(f"  [ablations] {type(e).__name__}: {e}")
    finally:
        sys.argv = old_argv
    out = Path("fixes/output")
    for fname in ["ablations_report.txt", "ablations.json"]:
        src = out / fname
        if src.exists():
            shutil.copy(src, out / f"{output_prefix}_{fname}")


@app.local_entrypoint()
def main():
    print("Running fixed activation extraction on Modal...")
    results = extract_fixed_activations.remote()

    for key, value in results.items():
        out = Path(key)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(value) if isinstance(value, bytes) else out.write_text(value)
        print(f"Saved: {key}")

    for label in ["sabotaged", "fixed"]:
        report = Path(f"fixes/output/{label}_ablations_report.txt")
        if report.exists():
            print(f"\n{'='*60}\nABLATIONS — {label.upper()}\n{'='*60}")
            print(report.read_text())
