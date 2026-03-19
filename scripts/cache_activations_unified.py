#!/usr/bin/env python3
"""
Cache neural activations for MMLU questions - UNIFIED VERSION.

This script extracts activations for ALL supported layers and ALL cached positions
in ONE forward pass, making layer/position sweeps dramatically faster.

This script:
1. Loads MMLU questions
2. Generates completions via VLLM API servers
3. Extracts neural activations for ALL layers × ALL positions in one pass
4. Saves unified activations cache to disk

Requires VLLM servers to be running (see vllm_launcher.sh).
"""

import os
import json
import argparse
import multiprocessing as mp
import numpy as np

from config import config
from src.data import load_mmlu_data, compute_correctness_labels
from src.generation import generate_with_vllm_multi_server
from src.activations import extract_all_layers_all_positions_multi_gpu


def cache_mmlu_activations_unified(
    prompt_name: str,
    model_name: str = None,
    layer_indices: list = None,
    positions_to_cache: list = None,
    num_examples: int = None,
    max_new_tokens: int = None,
    temperature: float = None,
    num_gpus: int = None,
    batch_size: int = None,
    mmlu_file: str = config.MMLU_FILE,
    filter_reliable: bool = False,
    reliable_questions_file: str = None
):
    """
    Cache layer activations for ALL layers and ALL positions in one unified cache.
    This runs ONE forward pass per sample and extracts from all layers/positions at once.

    Args:
        prompt_name: Name of the prompt to use (e.g., "benign", "semimalign") - REQUIRED
        model_name: Model name (defaults to config.MODEL_NAME)
        layer_indices: List of layers to cache (defaults to config.SUPPORTED_LAYERS)
        positions_to_cache: List of positions to cache (defaults to config.CACHED_POSITIONS)
        num_examples: Number of examples (defaults to config.DEFAULT_NUM_EXAMPLES)
        max_new_tokens: Max tokens to generate (defaults to config.MAX_NEW_TOKENS)
        temperature: Sampling temperature (defaults to config.TEMPERATURE)
        num_gpus: Number of GPUs (defaults to config.VLLM_NUM_SERVERS)
        batch_size: Batch size for activation extraction (defaults to config.ACTIVATION_BATCH_SIZE)
        mmlu_file: Path to MMLU data file
        filter_reliable: Whether to filter to only reliable questions (defaults to False)
        reliable_questions_file: Path to reliable questions JSON file (defaults to auto-detected path)
    """
    # Use config defaults if not specified
    model_name = model_name or config.MODEL_NAME
    layer_indices = layer_indices if layer_indices is not None else config.SUPPORTED_LAYERS
    positions_to_cache = positions_to_cache if positions_to_cache is not None else config.CACHED_POSITIONS
    num_examples = num_examples or config.DEFAULT_NUM_EXAMPLES
    max_new_tokens = max_new_tokens or config.MAX_NEW_TOKENS
    temperature = temperature if temperature is not None else config.TEMPERATURE
    num_gpus = num_gpus or config.VLLM_NUM_SERVERS
    batch_size = batch_size if batch_size is not None else config.ACTIVATION_BATCH_SIZE
    
    # Set default reliable questions file if filtering
    if filter_reliable and reliable_questions_file is None:
        reliable_questions_file = os.path.join(
            config.BASE_DIR, config.MODEL_SHORT_NAME, "reliable_questions.json"
        )

    # Create unified output directory
    cache_prompt_name = f"{prompt_name}_filtered" if filter_reliable else prompt_name
    filter_suffix = "filtered" if filter_reliable else "unfiltered"
    cache_name = f"unified_n{num_examples}_{filter_suffix}"
    output_dir = os.path.join(config.CACHED_ACTIVATIONS_DIR, cache_prompt_name, cache_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("UNIFIED ACTIVATION CACHING")
    print("=" * 80)
    config.print_config()
    print(f"\nRun Parameters:")
    print(f"  Prompt: {prompt_name}")
    print(f"  Filtered: {filter_reliable}")
    if filter_reliable:
        print(f"  Reliable Questions File: {reliable_questions_file}")
    print(f"  Layers: {layer_indices} ({len(layer_indices)} layers)")
    print(f"  Positions: {positions_to_cache} ({len(positions_to_cache)} positions)")
    print(f"  Examples: {num_examples}")
    print(f"  Max New Tokens: {max_new_tokens}")
    print(f"  Temperature: {temperature}")
    print(f"  Batch Size (activation): {batch_size}")
    print(f"  Output: {output_dir}")
    print("=" * 80)

    # Load MMLU data (with optional filtering)
    print(f"\nLoading {num_examples} MMLU examples from {mmlu_file}...")
    questions, prompts = load_mmlu_data(
        mmlu_file, num_examples, 
        prompt_name=prompt_name,
        filter_reliable=filter_reliable,
        reliable_questions_file=reliable_questions_file
    )
    print(f"Loaded {len(prompts)} samples")

    # Step 1: Generate completions via VLLM (with caching)
    print(f"\n{'=' * 80}")
    print(f"STEP 1: Generating completions with VLLM servers (or loading from cache)")
    print(f"{'=' * 80}")
    generated_texts, completions_only = generate_with_vllm_multi_server(
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        model_name=model_name,
        num_servers=num_gpus,
        base_port=config.VLLM_BASE_PORT,
        max_concurrent_requests=config.MAX_CONCURRENT_REQUESTS_PER_SERVER,
        use_cache=True,
        model_short_name=config.MODEL_SHORT_NAME,
        prompt_name=cache_prompt_name,
        subject_type="all"
    )

    # Step 2: Extract activations for ALL layers × ALL positions in ONE pass
    print(f"\n{'=' * 80}")
    print(f"STEP 2: Extracting activations (UNIFIED - all layers × all positions)")
    print(f"{'=' * 80}")
    layer_position_activations, probed_tokens_by_position = extract_all_layers_all_positions_multi_gpu(
        model_name=model_name,
        full_texts=generated_texts,
        layer_indices=layer_indices,
        positions_to_extract=positions_to_cache,
        num_gpus=num_gpus,
        batch_size=batch_size,
        use_model_cache=True,
        gpu_ids=config.ACTIVATION_GPUS,
        completions=completions_only
    )

    print(f"\n{'=' * 80}")
    print(f"Unified activations extracted!")
    print(f"  Samples: {num_examples}")
    print(f"  Layers: {len(layer_indices)}")
    print(f"  Positions: {len(positions_to_cache)}")
    for layer_idx in layer_indices[:3]:
        for position in positions_to_cache[:2]:
            shape = layer_position_activations[layer_idx][position].shape
            print(f"  Layer {layer_idx}, position '{position}': {shape}")
    if len(layer_indices) > 3 or len(positions_to_cache) > 2:
        print(f"  ... (more layers/positions)")
    print(f"{'=' * 80}")

    # Compute correctness labels
    print(f"\nComputing correctness labels...")
    labels, predicted_answers, correct_answer_indices = compute_correctness_labels(
        questions, generated_texts
    )
    labels = np.array(labels)

    num_correct = np.sum(labels)
    num_incorrect = len(labels) - num_correct
    accuracy = num_correct / len(labels) if len(labels) > 0 else 0

    print(f"  Correct: {num_correct}/{len(labels)} ({100 * accuracy:.1f}%)")
    print(f"  Incorrect: {num_incorrect}/{len(labels)}")

    # Extract subjects
    subjects = np.array([q['subject'] for q in questions])

    # Save unified cache
    print(f"\nSaving unified cache...")
    
    # Save layer/position activations in a structured way
    for layer_idx in layer_indices:
        for position in positions_to_cache:
            filename = f"activations_layer{layer_idx:02d}_pos-{position}.npy"
            filepath = os.path.join(output_dir, filename)
            np.save(filepath, layer_position_activations[layer_idx][position])
            print(f"  ✓ {filename}")
    
    # Save probed tokens for each position
    for position in positions_to_cache:
        data = np.array(probed_tokens_by_position[position], dtype=object)
        
        # Save as .npy
        filename = f"probed_tokens_pos-{position}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, data)
        print(f"  ✓ {filename}")
        
        # Also save as .json
        json_filename = f"probed_tokens_pos-{position}.json"
        json_filepath = os.path.join(output_dir, json_filename)
        with open(json_filepath, 'w') as f:
            json.dump(data.tolist(), f, indent=2)
        print(f"  ✓ {json_filename}")
    
    # Save shared data (same for all layer/position combos)
    shared_files = {
        "labels.npy": labels,
        "subjects.npy": subjects,
        "prompts.npy": np.array(prompts, dtype=object),
        "generated.npy": np.array(generated_texts, dtype=object),
        "completions.npy": np.array(completions_only, dtype=object),
        "predicted_answers.npy": np.array(predicted_answers, dtype=object),
        "correct_answer_indices.npy": np.array(correct_answer_indices),
    }
    
    for filename, data in shared_files.items():
        # Save as .npy
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, data)
        print(f"  ✓ {filename}")
        
        # Also save as .json
        json_filename = filename.replace('.npy', '.json')
        json_filepath = os.path.join(output_dir, json_filename)
        json_data = data.tolist() if isinstance(data, np.ndarray) else data
        with open(json_filepath, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ {json_filename}")
    
    # Save questions as JSON
    questions_file = os.path.join(output_dir, "questions.json")
    with open(questions_file, 'w') as f:
        json.dump(questions, f, indent=2)
    print(f"  ✓ questions.json")
    
    # Compute probed token statistics for each position
    probed_token_stats = {}
    for position in positions_to_cache:
        tokens = probed_tokens_by_position[position]
        if tokens and any(tokens):
            from collections import Counter
            token_counts = Counter(tokens)
            probed_token_stats[position] = {
                "unique_tokens": len(token_counts),
                "most_common": token_counts.most_common(10),
                "total": len(tokens)
            }
    
    # Save metadata
    metadata = {
        "cache_type": "unified",
        "model_name": model_name,
        "model_short_name": config.MODEL_SHORT_NAME,
        "prompt_name": prompt_name,
        "cache_prompt_name": cache_prompt_name,
        "filter_reliable": filter_reliable,
        "reliable_questions_file": reliable_questions_file if filter_reliable else None,
        "layer_indices": layer_indices,
        "positions_cached": positions_to_cache,
        "num_examples": num_examples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "num_correct": int(num_correct),
        "num_incorrect": int(num_incorrect),
        "accuracy": float(accuracy),
        "probed_token_stats": probed_token_stats,
        "description": f"Unified MMLU activations for {len(layer_indices)} layers × {len(positions_to_cache)} positions, prompt '{prompt_name}'{' (filtered to reliable questions)' if filter_reliable else ''}. Labels: 1=correct, 0=incorrect.",
        "file_structure": {
            "activations": "activations_layer{XX}_pos-{position}.npy",
            "probed_tokens": "probed_tokens_pos-{position}.npy",
            "shared": list(shared_files.keys())
        }
    }
    metadata_file = os.path.join(output_dir, "metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ metadata.json")

    print(f"\n{'=' * 80}")
    print(f"✓ Unified caching complete!")
    print(f"  Cached {len(prompts)} samples × {len(layer_indices)} layers × {len(positions_to_cache)} positions")
    print(f"  Accuracy: {100 * accuracy:.1f}%")
    print(f"  Location: {output_dir}")
    print(f"  Total activations files: {len(layer_indices) * len(positions_to_cache)}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # Set multiprocessing start method for CUDA compatibility
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    parser = argparse.ArgumentParser(description="Cache MMLU activations (unified)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt name (e.g., 'benign', 'semimalign')")
    parser.add_argument("--layers", type=int, nargs='+', default=None,
                        help=f"Layer indices to cache (default: config.SUPPORTED_LAYERS)")
    parser.add_argument("--positions", type=str, nargs='+', default=None,
                        help=f"Positions to cache (default: config.CACHED_POSITIONS)")
    parser.add_argument("--num-examples", type=int, default=None,
                        help="Number of examples")
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="Number of GPUs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for activation extraction")
    parser.add_argument("--mmlu-file", type=str, default=config.MMLU_FILE,
                        help="Path to MMLU data file")
    parser.add_argument("--filtered", action="store_true",
                        help="Filter to only reliable questions")
    parser.add_argument("--reliable-questions-file", type=str, default=None,
                        help="Path to reliable questions JSON file")

    args = parser.parse_args()

    cache_mmlu_activations_unified(
        prompt_name=args.prompt,
        layer_indices=args.layers,
        positions_to_cache=args.positions,
        num_examples=args.num_examples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        num_gpus=args.num_gpus,
        batch_size=args.batch_size,
        mmlu_file=args.mmlu_file,
        filter_reliable=args.filtered,
        reliable_questions_file=args.reliable_questions_file
    )

