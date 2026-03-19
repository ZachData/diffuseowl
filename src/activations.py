"""
Neural activation extraction utilities.
"""
import os
import time
start_time = time.time()

print("importing in activations.py")
import multiprocessing as mp
from multiprocessing import Manager
from typing import List, Literal, Optional, Tuple, TYPE_CHECKING
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import config

# if TYPE_CHECKING:
#     import torch
#     from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

print(f"done importing in activations.py ({time.time() - start_time:.2f}s)")

TokenPosition = Literal["last", "first", "middle", "all", "all_appended", "letter", "letter+1", "letter-1", "yes_no", "yes_no-1"]

ENSURE_MODEL_CACHED = False

def ensure_model_cached(model_name: str):
    """
    Pre-download model to cache to avoid multiple processes downloading simultaneously.

    Args:
        model_name: Model name to cache
    """
    # Lazy import
    from transformers import AutoConfig
    
    try:
        print(f"Ensuring {model_name} is cached locally...")
        # This will download/cache the model config and weights if not already present
        AutoConfig.from_pretrained(model_name)
        print(f"  ✓ Model cached")
    except Exception as e:
        print(f"  ⚠ Warning: Could not pre-cache model: {e}")
        print(f"  Processes will download individually (slower)")


def find_letter_token_position(
    completion: str,
    tokenizer,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the token position where the answer letter appears in the completion.
    
    Args:
        completion: Just the completion part
        tokenizer: The tokenizer
        
    Returns:
        Tuple of (token_position, literal_token_string) or (None, None) if not found
    """
    # Tokenize the completion
    completion_tokens = tokenizer.tokenize(completion)
    # Find the index of ['\\', 'box', '{'] or ['_\\', 'box', '{']
    for i in range(len(completion_tokens)-4): # the minimum valid box is e.g. \box{A}, which is 5 tokens
        if '\\' in completion_tokens[i] and completion_tokens[i+1:i+3] == ['box', '{']:
            index = i+3 # the index of the answer letter
            return index, completion_tokens[index]
    return None, None


def find_yes_no_token_position(
    tokenizer,
    input_ids: "torch.Tensor"
) -> Optional[int]:
    """
    Find the token position where "yes" or "no" appears after "Assistant:" *in the full text*.
    
    This is specifically for lie detector experiments where the model responds yes/no
    to "Did you give the correct answer?"
    
    Args:
        tokenizer: The tokenizer
        input_ids: Token IDs for the full text
    
    Returns:
        Token position index, or None if not found
    """
    tokenized = [tokenizer.decode(x, skip_special_tokens=False) for x in input_ids]
    
    # Find "Assistant:" and then look for yes/no shortly after
    found_assistant = False
    for i in range(len(tokenized)):
        token_lower = tokenized[i].lower().strip()
        
        if not found_assistant and "assistant" in token_lower:
            found_assistant = True
            continue
        
        if found_assistant:
            # Check if this token contains yes or no
            # Common patterns: "yes", "Yes", " yes", "no", "No", " no"
            if "yes" in token_lower or "no" in token_lower:
                return i
    
    return None


def get_probe_positions(
    input_ids: "torch.Tensor",
    special_token_ids: set,
    token_position: TokenPosition = "last"
) -> List[int]:
    """
    Get token positions to extract activations from.

    Args:
        input_ids: Token IDs (1D tensor)
        special_token_ids: Set of special token IDs to exclude
        token_position: Which position(s) to extract
            - "last": Last non-special token
            - "first": First non-special token
            - "middle": Middle non-special token
            - "all": All non-special tokens
            - "all_appended": All non-special tokens appended together

    Returns:
        List of token positions to extract
    """
    # Find all non-special token positions
    non_special_positions = [
        idx for idx, token_id in enumerate(input_ids.tolist())
        if int(token_id) not in special_token_ids
    ]

    if len(non_special_positions) == 0:
        # Fallback to last position if all are special tokens
        return [len(input_ids) - 1 if len(input_ids) > 0 else 0]

    if token_position == "last":
        return [non_special_positions[-1]]
    elif token_position == "first":
        return [non_special_positions[0]]
    elif token_position == "middle":
        mid_idx = len(non_special_positions) // 2
        return [non_special_positions[mid_idx]]
    elif token_position == "all":
        return non_special_positions
    elif token_position == "all_appended":
        return non_special_positions
    else:
        raise ValueError(f"Unknown token_position: {token_position}")


def extract_all_layers_all_positions_single_gpu(
    gpu_id: int,
    model_name: str,
    full_texts: List[str],
    layer_indices: List[int],
    positions_to_extract: List[str],
    return_dict: dict,
    index: int,
    batch_size: int = 4,
    use_cache: bool = True,
    completions: Optional[List[str]] = None
):
    """
    Extract activations from ALL specified layers and ALL specified positions in one forward pass.
    This is much more efficient than running forward passes for each layer separately.
    
    Args:
        gpu_id: GPU device ID
        model_name: Model name/path
        full_texts: List of texts to process
        layer_indices: List of layer indices to extract from (e.g., [0, 10, 20, 30, 47])
        positions_to_extract: List of positions to extract (e.g., ["last", "first", "middle", "letter"])
        return_dict: Shared dict for results
        index: Index in return dict
        batch_size: Process this many texts at once
        use_cache: Use cached model files
        completions: List of completion texts (required for "letter" position)
        
    Returns (via return_dict):
        Dictionary with:
            - 'activations': Dict[layer_idx][position] = np.array (num_samples, hidden_dim)
            - 'probed_tokens': Dict[position] = List[str] (which literal tokens were probed)
    """
    try:
        start_time = time.time()
        print(f"[GPU {gpu_id}] Loading model for UNIFIED activation extraction ({len(full_texts)} texts)...")
        print(f"[GPU {gpu_id}] Layers: {layer_indices}")
        print(f"[GPU {gpu_id}] Positions: {positions_to_extract}")
        
        # Set device
        device = f"cuda:{gpu_id}"
        
        # Load tokenizer and model
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        load_start = time.time()
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map=device,
            attn_implementation="eager",
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        print(f"[GPU {gpu_id}] Model loaded in {time.time() - load_start:.2f}s")
        
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.eval()
        
        # Initialize storage: {layer_idx: {position: []}}
        layer_position_activations = {
            layer_idx: {pos: [] for pos in positions_to_extract}
            for layer_idx in layer_indices
        }
        probed_tokens_by_position = {pos: [] for pos in positions_to_extract}
        
        if "letter" in positions_to_extract and completions is None:
            raise ValueError("completions must be provided when extracting 'letter' position")
        if "yes_no" in positions_to_extract and completions is None:
            raise ValueError("completions must be provided when extracting 'yes_no' position")
        
        print(f"[GPU {gpu_id}] Extracting activations (batch_size={batch_size})...")
        
        with torch.no_grad():
            for batch_start in range(0, len(full_texts), batch_size):
                batch_end = min(batch_start + batch_size, len(full_texts))
                batch_texts = full_texts[batch_start:batch_end]
                
                if (batch_end) % 20 == 0 or batch_end == len(full_texts):
                    elapsed = time.time() - start_time
                    rate = batch_end / elapsed if elapsed > 0 else 0
                    print(f"  [GPU {gpu_id}] Processing {batch_end}/{len(full_texts)} ({rate:.1f} texts/sec)")
                
                for idx, full_text in enumerate(batch_texts):
                    text_idx = batch_start + idx
                    
                    # Tokenize
                    inputs = tokenizer(full_text, return_tensors="pt").to(device)
                    input_ids = inputs.input_ids[0]
                    special_token_ids = set(getattr(tokenizer, "all_special_ids", []))
                    
                    # Run ONE forward pass to get ALL hidden states
                    forward_outputs = model(**inputs, output_hidden_states=True, use_cache=False)
                    
                    # For each position, find the token index(es) to extract
                    position_to_token_indices = {}
                    for position in positions_to_extract:
                        if position in ["letter", "letter+1", "letter-1"]:
                            completion = completions[text_idx] if completions else ""
                            letter_pos, probed_token_str = find_letter_token_position(
                                completion, tokenizer
                            )
                            if letter_pos is not None:
                                if position == "letter+1":
                                    letter_pos += 1
                                elif position == "letter-1":
                                    letter_pos -= 1
                                probed_token_str = tokenizer.decode([input_ids[letter_pos]], skip_special_tokens=False)
                                position_to_token_indices[position] = [letter_pos]
                                if text_idx < len(full_texts):  # Only record once
                                    if len(probed_tokens_by_position[position]) == text_idx:
                                        probed_tokens_by_position[position].append(probed_token_str)
                            else:
                                # Fallback to last
                                if config.VERBOSITY >= 2:
                                    print(f"Warning: No letter token found for {full_text}, falling back to last token")
                                fallback_positions = get_probe_positions(input_ids, special_token_ids, "last")
                                position_to_token_indices[position] = fallback_positions
                                if len(probed_tokens_by_position[position]) == text_idx:
                                    probed_tokens_by_position[position].append(
                                        tokenizer.decode([input_ids[fallback_positions[0]]], skip_special_tokens=False)
                                    )
                        elif position in ["yes_no", "yes_no-1"]:
                            completion = completions[text_idx] if completions else ""
                            yes_no_pos = find_yes_no_token_position(
                                tokenizer, input_ids
                            )
                            if yes_no_pos is not None:
                                if position == "yes_no-1":
                                    yes_no_pos -= 1
                                probed_token_str = tokenizer.decode([input_ids[yes_no_pos]], skip_special_tokens=False)
                                position_to_token_indices[position] = [yes_no_pos]
                                if text_idx < len(full_texts):  # Only record once
                                    if len(probed_tokens_by_position[position]) == text_idx:
                                        probed_tokens_by_position[position].append(probed_token_str)
                            else:
                                # Fallback to last
                                if config.VERBOSITY >= 2:
                                    print(f"Warning: No yes_no token found for {full_text}, falling back to last token")
                                fallback_positions = get_probe_positions(input_ids, special_token_ids, "last")
                                position_to_token_indices[position] = fallback_positions
                                if len(probed_tokens_by_position[position]) == text_idx:
                                    probed_tokens_by_position[position].append(
                                        tokenizer.decode([input_ids[fallback_positions[0]]], skip_special_tokens=False)
                                    )
                        else:
                            position_to_token_indices[position] = get_probe_positions(
                                input_ids, special_token_ids, position
                            )
                            # Record probed token for single-token positions
                            if position not in ["all", "all_appended"] and len(probed_tokens_by_position[position]) == text_idx:
                                token_str = tokenizer.decode([input_ids[position_to_token_indices[position][0]]], skip_special_tokens=False)
                                probed_tokens_by_position[position].append(token_str)
                    
                    # Extract from each layer
                    for layer_idx in layer_indices:
                        # hidden_states[0] is embeddings, hidden_states[1] is layer 0, etc.
                        layer_output = forward_outputs.hidden_states[layer_idx + 1]
                        
                        # Extract for each position
                        for position in positions_to_extract:
                            probe_positions = position_to_token_indices[position]
                            
                            if position == "all" or position == "all_appended":
                                activations_at_positions = [
                                    layer_output[0, pos, :].float().cpu().numpy()
                                    for pos in probe_positions
                                ]
                                if position == "all_appended":
                                    final_activation = np.concatenate(activations_at_positions, axis=0)
                                else:
                                    final_activation = np.mean(activations_at_positions, axis=0)
                            else:
                                # Single position
                                pos = probe_positions[0]
                                final_activation = layer_output[0, pos, :].float().cpu().numpy()
                            
                            layer_position_activations[layer_idx][position].append(final_activation)
                
                # Clear CUDA cache periodically
                if batch_end % 50 == 0:
                    torch.cuda.empty_cache()
        
        # Convert lists to numpy arrays
        result_activations = {}
        for layer_idx in layer_indices:
            result_activations[layer_idx] = {}
            for position in positions_to_extract:
                result_activations[layer_idx][position] = np.array(
                    layer_position_activations[layer_idx][position]
                )
        
        total_time = time.time() - start_time
        rate = len(full_texts) / total_time if total_time > 0 else 0
        print(f"[GPU {gpu_id}] Complete! {len(full_texts)} texts × {len(layer_indices)} layers × {len(positions_to_extract)} positions in {total_time:.2f}s ({rate:.1f} texts/sec)")
        
        return_dict[index] = {
            'activations': result_activations,
            'probed_tokens': probed_tokens_by_position
        }
        
    except Exception as e:
        print(f"[GPU {gpu_id}] Error: {e}")
        import traceback
        traceback.print_exc()
        return_dict[index] = {
            'activations': {},
            'probed_tokens': {}
        }


def extract_all_layers_all_positions_multi_gpu(
    model_name: str,
    full_texts: List[str],
    layer_indices: List[int],
    positions_to_extract: List[str],
    num_gpus: int = 8,
    batch_size: int = 4,
    use_model_cache: bool = True,
    gpu_ids: List[int] = None,
    completions: Optional[List[str]] = None
) -> Tuple[dict, dict]:
    """
    Extract activations from ALL specified layers and ALL specified positions using multiple GPUs.
    Runs ONE forward pass per text to extract everything at once.
    
    Args:
        model_name: Name of the model
        full_texts: List of full generated texts
        layer_indices: List of layer indices to extract from
        positions_to_extract: List of positions to extract
        num_gpus: Number of GPUs to use
        batch_size: Batch size for processing
        use_model_cache: Use cached model files
        gpu_ids: Specific GPU IDs to use
        completions: List of completion texts (required for "letter" position)
        
    Returns:
        Tuple of (layer_position_activations, probed_tokens_by_position):
            - layer_position_activations: Dict[layer_idx][position] = np.array (num_samples, hidden_dim)
            - probed_tokens_by_position: Dict[position] = List[str]
    """
    # Determine which GPUs to use
    if gpu_ids is None:
        gpu_ids = list(range(num_gpus))
    else:
        num_gpus = len(gpu_ids)
    
    print(f"\n{'='*60}")
    print(f"UNIFIED ACTIVATION EXTRACTION")
    print(f"Distributing {len(full_texts)} texts across {num_gpus} GPUs")
    print(f"GPU IDs: {gpu_ids}")
    print(f"Layers: {layer_indices} ({len(layer_indices)} layers)")
    print(f"Positions: {positions_to_extract} ({len(positions_to_extract)} positions)")
    print(f"Batch size: {batch_size}")
    print(f"{'='*60}")
    
    # Pre-cache model
    if use_model_cache and ENSURE_MODEL_CACHED:
        ensure_model_cached(model_name)
    
    # Split texts and completions into chunks
    chunk_size = (len(full_texts) + num_gpus - 1) // num_gpus
    text_chunks = [full_texts[i:i + chunk_size] for i in range(0, len(full_texts), chunk_size)]
    
    completion_chunks = None
    if completions is not None:
        completion_chunks = [completions[i:i + chunk_size] for i in range(0, len(completions), chunk_size)]
    
    print(f"Chunk sizes: {[len(chunk) for chunk in text_chunks]}")
    
    # Use Manager to share results
    manager = Manager()
    return_dict = manager.dict()
    
    if num_gpus == 1:
        extract_all_layers_all_positions_single_gpu(
            gpu_ids[0], model_name, text_chunks[0], layer_indices, positions_to_extract,
            return_dict, 0, batch_size, use_model_cache,
            completion_chunks[0] if completion_chunks else None
        )
    else:
        processes = []
        for i, chunk in enumerate(text_chunks):
            gpu_id = gpu_ids[i]
            completion_chunk = completion_chunks[i] if completion_chunks else None
            p = mp.Process(
                target=extract_all_layers_all_positions_single_gpu,
                args=(gpu_id, model_name, chunk, layer_indices, positions_to_extract,
                      return_dict, i, batch_size, use_model_cache, completion_chunk)
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    
    # Combine results from all GPUs
    layer_position_activations = {
        layer_idx: {pos: [] for pos in positions_to_extract}
        for layer_idx in layer_indices
    }
    probed_tokens_by_position = {pos: [] for pos in positions_to_extract}
    
    for i in range(len(text_chunks)):
        if i in return_dict:
            chunk_result = return_dict[i]
            chunk_activations = chunk_result['activations']
            chunk_tokens = chunk_result['probed_tokens']
            
            # Combine activations
            for layer_idx in layer_indices:
                for position in positions_to_extract:
                    if layer_idx in chunk_activations and position in chunk_activations[layer_idx]:
                        layer_position_activations[layer_idx][position].append(
                            chunk_activations[layer_idx][position]
                        )
            
            # Combine probed tokens
            for position in positions_to_extract:
                if position in chunk_tokens:
                    probed_tokens_by_position[position].extend(chunk_tokens[position])
        else:
            print(f"Warning: GPU {i} did not return results")
    
    # Concatenate arrays
    for layer_idx in layer_indices:
        for position in positions_to_extract:
            if layer_position_activations[layer_idx][position]:
                layer_position_activations[layer_idx][position] = np.vstack(
                    layer_position_activations[layer_idx][position]
                )
            else:
                layer_position_activations[layer_idx][position] = np.array([])
    
    print(f"\nAll GPUs completed unified extraction!")
    print(f"  Samples: {len(full_texts)}")
    print(f"  Layers: {len(layer_indices)}")
    print(f"  Positions: {len(positions_to_extract)}")
    for layer_idx in layer_indices[:3]:  # Show first few layers
        for position in positions_to_extract:
            shape = layer_position_activations[layer_idx][position].shape
            print(f"  Layer {layer_idx}, position '{position}': {shape}")
    if len(layer_indices) > 3:
        print(f"  ... ({len(layer_indices) - 3} more layers)")
    
    return layer_position_activations, probed_tokens_by_position

