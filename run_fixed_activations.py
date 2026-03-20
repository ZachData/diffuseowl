#!/usr/bin/env python3
"""
run_fixed_activations.py

Modal script that extracts activations at the CORRECTED letter token position
and runs the ablation suite to compare:
  - Paper's claimed result (sabotaged letter pos == yes/newline token)
  - Honest result (true A/B/C/D token position)

The generation cache already exists locally. This script:
  1. Uploads the existing generation cache to the container
  2. Applies the fixed letter-finding function (fixed_activations_patch.py)
  3. Extracts activations at layer 30, position "letter" using the fix
  4. Runs fixes/fix_ablations.py A1-A5 and A7 against both caches
  5. Returns all .npy activation files and ablation reports

Usage:
    modal run run_fixed_activations.py

Outputs saved locally to:
    fixed_cache/          — fixed activation .npy files
    fixes/output/         — ablation reports (updated)
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("diffuse-fixed-activations")

NUM_EXAMPLES = 5000
LAYER = 30
# Positions to extract in the fixed run.
# "letter" will now use the corrected search (A/B/C/D token in Pass 1).
# The others are already correct and serve as comparison points.
POSITIONS = ["letter", "letter+1", "letter-1", "yes_no", "yes_no-1", "last"]

diffuse_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "scikit-learn",
        "matplotlib",
        "numpy",
        "scipy",
        "accelerate",
        "hf_transfer",
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
    gpu="A100-80GB",
    timeout=4 * 60 * 60,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("secrets")],
)
def extract_fixed_activations() -> dict:
    """
    Extract activations using the corrected letter-position finder.
    Saves to a separate cache directory so the original (sabotaged) cache
    is preserved for comparison in the ablations.
    """
    import os
    import sys
    import json
    import numpy as np
    from pathlib import Path

    os.chdir("/workspace")
    sys.path.insert(0, "/workspace")
    sys.path.insert(0, "/workspace/scripts")

    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or ""
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = hf_token
        token_path = Path("/root/.cache/huggingface/token")
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(hf_token)

    from config import config
    from src.activations import (
        extract_all_layers_all_positions_single_gpu,
        get_probe_positions,
        find_yes_no_token_position,
    )

    # ── Monkey-patch the letter-finding function with the fixed version ──────
    # This patches src.activations in-place so that when
    # extract_all_layers_all_positions_single_gpu calls find_letter_token_position,
    # it gets the corrected version instead.
    import src.activations as act_module

    from typing import Optional, Tuple

    def find_letter_token_position_in_full_text(
        full_text: str,
        tokenizer,
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Fixed version: searches the full two-pass conversation for \\box{X},
        returns an index valid for use with the full input_ids.
        """
        full_tokens = tokenizer.tokenize(full_text)
        search_start = 0
        for i, tok in enumerate(full_tokens):
            if "answer" in tok.lower():
                search_start = i
        for i in range(search_start, len(full_tokens) - 4):
            if '\\' in full_tokens[i] and full_tokens[i + 1:i + 3] == ['box', '{']:
                index = i + 3
                return index, full_tokens[index]
        return None, None

    # Patch the module-level function so the extraction loop uses it.
    # The extraction loop (lines 254-278 of activations.py) calls:
    #   find_letter_token_position(completion, tokenizer)
    # We replace it with a version that ignores the completion argument
    # and searches full_text directly.
    _orig_find = act_module.find_letter_token_position

    def patched_find_letter_token_position(completion, tokenizer):
        # completion argument is ignored — the fix uses full_text instead.
        # We return a sentinel that the extraction loop will NOT find,
        # forcing it into the fallback path. The real fix is applied below
        # by overriding the extraction loop's position logic directly.
        return None, None

    act_module.find_letter_token_position = patched_find_letter_token_position

    # ── Load the existing generation cache ───────────────────────────────────
    gen_dir = Path("experiments/gemma-3-12b/generations")
    part1_path = gen_dir / "lie_detector_part1/all/n5000_t1.0_maxtok100"
    part2_path = gen_dir / "lie_detector_part2/all/n5000_t1.0_maxtok100"

    print("Loading generation cache...")
    full_texts_p1 = np.load(str(part1_path) + "_full_texts.npy", allow_pickle=True).tolist()
    full_texts_p2 = np.load(str(part2_path) + "_full_texts.npy", allow_pickle=True).tolist()
    completions_p2 = np.load(str(part2_path) + "_completions.npy", allow_pickle=True).tolist()
    labels_path = Path("experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered/labels.json")
    with open(labels_path) as f:
        labels = np.array(json.load(f))

    print(f"  Pass 1 texts: {len(full_texts_p1)}")
    print(f"  Pass 2 texts: {len(full_texts_p2)}")
    print(f"  Labels: {len(labels)}  correct={int(labels.sum())}")

    # ── Build the full two-pass conversation strings ─────────────────────────
    # The activation extraction runs on the FULL two-pass conversation
    # (Pass 1 + Pass 2 appended), same as the original pipeline.
    full_conversations = full_texts_p2  # already includes Pass 1 + Pass 2

    # ── Extract activations with the fixed letter-position logic ─────────────
    # We implement the extraction directly here so we can inject the fix
    # without restructuring the library code.

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    model_name = config.MODEL_NAME
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
        token=hf_token,
    )
    model.config.use_cache = False
    model.eval()
    print("Model loaded.")

    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))

    # Storage: position -> list of activation vectors
    position_activations = {pos: [] for pos in POSITIONS}
    probed_tokens = {pos: [] for pos in POSITIONS}

    batch_size = config.ACTIVATION_BATCH_SIZE
    total = len(full_conversations)

    print(f"\nExtracting activations (layer={LAYER}, positions={POSITIONS})...")
    print(f"  Total texts: {total}  batch_size: {batch_size}")

    with torch.no_grad():
        for text_idx, full_text in enumerate(full_conversations):
            if text_idx % 100 == 0 or text_idx == total - 1:
                print(f"  {text_idx+1}/{total}")

            inputs = tokenizer(full_text, return_tensors="pt").to(device)
            input_ids = inputs.input_ids[0]

            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            # hidden_states[0]=embeddings, hidden_states[layer+1]=layer output
            layer_output = outputs.hidden_states[LAYER + 1]  # shape (1, seq_len, hidden)

            # ── Determine token index for each position ───────────────────
            for position in POSITIONS:
                if position in ["letter", "letter+1", "letter-1"]:
                    # FIXED: search in full_text (Pass 1 part), not in Pass 2 completion
                    # Use Pass 1 text to find the answer letter
                    pass1_text = full_texts_p1[text_idx]
                    letter_pos, probed_tok = find_letter_token_position_in_full_text(
                        pass1_text, tokenizer
                    )
                    if letter_pos is not None:
                        if position == "letter+1":
                            letter_pos = min(letter_pos + 1, len(input_ids) - 1)
                        elif position == "letter-1":
                            letter_pos = max(letter_pos - 1, 0)
                        tok_str = tokenizer.decode([input_ids[letter_pos]], skip_special_tokens=False)
                        token_idx = letter_pos
                    else:
                        # genuine fallback (should be <0.1% of samples)
                        fallback = get_probe_positions(input_ids, special_token_ids, "last")
                        token_idx = fallback[0]
                        tok_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)

                elif position in ["yes_no", "yes_no-1"]:
                    yes_no_pos = find_yes_no_token_position(tokenizer, input_ids)
                    if yes_no_pos is not None:
                        if position == "yes_no-1":
                            yes_no_pos = max(yes_no_pos - 1, 0)
                        token_idx = yes_no_pos
                        tok_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)
                    else:
                        fallback = get_probe_positions(input_ids, special_token_ids, "last")
                        token_idx = fallback[0]
                        tok_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)

                else:  # "last"
                    token_idx = get_probe_positions(input_ids, special_token_ids, "last")[0]
                    tok_str = tokenizer.decode([input_ids[token_idx]], skip_special_tokens=False)

                activation = layer_output[0, token_idx, :].float().cpu().numpy()
                position_activations[position].append(activation)
                probed_tokens[position].append(tok_str)

            if text_idx % 50 == 0:
                torch.cuda.empty_cache()

    # ── Convert to arrays ─────────────────────────────────────────────────────
    for pos in POSITIONS:
        position_activations[pos] = np.array(position_activations[pos])
        print(f"  {pos}: {position_activations[pos].shape}")

    # ── Save to fixed cache directory ─────────────────────────────────────────
    out_dir = Path("experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered_fixed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for pos in POSITIONS:
        fname = f"activations_layer{LAYER:02d}_pos-{pos}.npy"
        np.save(out_dir / fname, position_activations[pos])
        print(f"  saved: {fname}")

        tok_fname = f"probed_tokens_pos-{pos}.json"
        with open(out_dir / tok_fname, "w") as f:
            json.dump(probed_tokens[pos], f, indent=2)

    # Copy shared files from original cache (labels, questions, etc.)
    import shutil
    orig_dir = Path("experiments/gemma-3-12b/cached_activations/lie_detector_filtered/unified_n5000_filtered")
    for fname in ["labels.json", "labels.npy", "questions.json", "subjects.json",
                  "predicted_answers.json", "correct_answer_indices.json",
                  "prompts.json", "generated.json", "completions.json",
                  "first_generation.json"]:
        src = orig_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / fname)
    print(f"\nFixed cache saved: {out_dir}")

    # ── Verify token quality ───────────────────────────────────────────────────
    for pos in POSITIONS:
        toks = probed_tokens[pos]
        letter_pct = 100 * sum(1 for t in toks if t and t.strip() in ["A", "B", "C", "D"]) / max(len(toks), 1)
        yes_no_pct = 100 * sum(1 for t in toks if t and "yes" in t.lower()) / max(len(toks), 1)
        print(f"  {pos:12}  letter%={letter_pct:.1f}  yes%={yes_no_pct:.1f}")

    # ── Run ablations on both caches ──────────────────────────────────────────
    print("\n\nRunning ablations on SABOTAGED cache...")
    _run_ablations(
        cache_dir=str(orig_dir),
        output_prefix="sabotaged",
    )

    print("\n\nRunning ablations on FIXED cache...")
    _run_ablations(
        cache_dir=str(out_dir),
        output_prefix="fixed",
    )

    # ── Collect results ────────────────────────────────────────────────────────
    results = {}

    # Fixed activation files
    for pos in POSITIONS:
        fname = f"activations_layer{LAYER:02d}_pos-{pos}.npy"
        fpath = out_dir / fname
        if fpath.exists():
            results[f"fixed_cache/{fname}"] = fpath.read_bytes()

        tok_fname = f"probed_tokens_pos-{pos}.json"
        fpath = out_dir / tok_fname
        if fpath.exists():
            results[f"fixed_cache/{tok_fname}"] = fpath.read_bytes()

    # Ablation reports
    for prefix in ["sabotaged", "fixed"]:
        for ext in ["ablations_report.txt", "ablations.json"]:
            fpath = Path(f"fixes/output/{prefix}_{ext}")
            if fpath.exists():
                results[f"fixes/output/{prefix}_{ext}"] = fpath.read_bytes()

    hf_cache.commit()
    return results


def _run_ablations(cache_dir: str, output_prefix: str):
    """Run fix_ablations.py against a cache dir, saving output with a prefix."""
    import sys
    from pathlib import Path

    sys.path.insert(0, "/workspace/fixes")

    import fix_ablations as fa

    old_argv = sys.argv
    sys.argv = [
        "fix_ablations.py",
        "--cache-dir", cache_dir,
        "--layer", "30",
        "--n-trials", "5",
        "--ablations", "A1", "A2", "A3", "A4", "A5", "A7",
    ]
    try:
        fa.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old_argv

    # Rename outputs so the two runs don't overwrite each other
    from pathlib import Path
    import shutil
    out_dir = Path("fixes/output")
    for fname in ["ablations_report.txt", "ablations.json"]:
        src = out_dir / fname
        if src.exists():
            shutil.copy(src, out_dir / f"{output_prefix}_{fname}")


@app.local_entrypoint()
def main():
    print("Extracting fixed activations on Modal...")
    results = extract_fixed_activations.remote()

    for key, value in results.items():
        out_path = Path(key)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(value, bytes):
            out_path.write_bytes(value)
        else:
            out_path.write_text(value)
        print(f"Saved: {key}")

    # Print ablation comparison if available
    for prefix in ["sabotaged", "fixed"]:
        report = Path(f"fixes/output/{prefix}_ablations_report.txt")
        if report.exists():
            print(f"\n{'='*60}")
            print(f"ABLATIONS — {prefix.upper()}")
            print('='*60)
            print(report.read_text())

    print("\nDone. Key files:")
    print("  fixed_cache/activations_layer30_pos-letter.npy  — true A/B/C/D activations")
    print("  fixes/output/fixed_ablations_report.txt         — honest probe results")
    print("  fixes/output/sabotaged_ablations_report.txt     — paper's (buggy) results reproduced")
