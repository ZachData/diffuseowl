"""
fixes/fixed_activations_patch.py

DROP-IN REPLACEMENT for the two sabotaged components in src/activations.py.

WHAT WAS WRONG:
  Bug 1 (pipeline wiring) — cache_lie_detector_activations.py line 194:
    completions=second_completions was passed to extraction.
    The letter-finder searched Pass 2 "yes" text for \\box{A}, found nothing,
    and fell back to the last token of the full conversation.

  Bug 2 (index space mismatch) — activations.py lines 256-265:
    find_letter_token_position() tokenizes only the completion and returns
    an index within that short token sequence. That index is then used
    directly to index into input_ids (the full two-pass conversation).
    These are different namespaces — a small index into a long sequence
    points to a completely unrelated token.

WHAT THE FIX DOES:
  Replaces find_letter_token_position(completion, tokenizer) with
  find_letter_token_position_in_full_text(full_text, tokenizer) which:
    - Tokenizes the full conversation text
    - Searches within those full input_ids for the \\box{X} pattern
    - Returns an index that is already in full-conversation token space
    - No separate completion argument needed; no index translation required

HOW TO APPLY:
  1. In src/activations.py, replace find_letter_token_position with
     find_letter_token_position_in_full_text (below).
  2. In src/activations.py lines 254-265, replace:
       completion = completions[text_idx]
       letter_pos, probed_token_str = find_letter_token_position(completion, tokenizer)
     with:
       letter_pos, probed_token_str = find_letter_token_position_in_full_text(
           full_text, tokenizer
       )
     (no completions argument needed for letter positions)
  3. In scripts/cache_lie_detector_activations.py line 194:
     completions= can remain for yes_no positions, but letter positions
     no longer need it since the fix works directly on full_texts.
"""

from typing import Optional, Tuple


def find_letter_token_position_in_full_text(
    full_text: str,
    tokenizer,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Find the token position of the answer letter (A/B/C/D inside \\box{X})
    within the FULL tokenized text (full two-pass conversation).

    This fixes two bugs in the original find_letter_token_position:

    Bug 1: The original took a separate 'completion' argument. In the lie
    detector pipeline, the Pass 2 completion ("yes") was passed — which
    contains no \\box{X} — causing systematic fallback to the last token.

    Bug 2: The original returned an index within the completion's own token
    sequence, which was then used directly as an index into input_ids (the
    full conversation). These are different token namespaces; the index
    pointed to a completely unrelated token.

    This function takes the full text, tokenizes it once, and returns an
    index that is already valid for use with those same input_ids.

    Args:
        full_text: The complete two-pass conversation string.
        tokenizer: The model tokenizer.

    Returns:
        Tuple of (token_index_in_full_text, literal_token_string)
        or (None, None) if \\box{X} is not found.
    """
    # Tokenize the full conversation — same tokenization used for input_ids
    full_tokens = tokenizer.tokenize(full_text)

    # Search for the \\box{X} pattern in full-conversation token space.
    # We anchor to the first occurrence after the "Your answer:" marker to
    # avoid matching any \\box{A} examples that may appear in the prompt itself.
    #
    # Strategy: find "answer" token as a soft anchor, then search forward.
    # If no anchor found, fall back to scanning the entire sequence.
    search_start = 0
    for i, tok in enumerate(full_tokens):
        if "answer" in tok.lower():
            search_start = i
            # Don't break — take the LAST "answer" occurrence, which
            # corresponds to "Your answer:" at the end of Pass 1.

    for i in range(search_start, len(full_tokens) - 4):
        if '\\' in full_tokens[i] and full_tokens[i + 1:i + 3] == ['box', '{']:
            index = i + 3  # index of the letter token (A/B/C/D)
            return index, full_tokens[index]

    return None, None


# ---------------------------------------------------------------------------
# Patch instructions for src/activations.py
# ---------------------------------------------------------------------------
#
# ORIGINAL (sabotaged) code at lines 254-278:
#
#   if position in ["letter", "letter+1", "letter-1"]:
#       completion = completions[text_idx] if completions else ""
#       letter_pos, probed_token_str = find_letter_token_position(
#           completion, tokenizer                          # ← Bug 1 + Bug 2
#       )
#       if letter_pos is not None:
#           ...
#           probed_token_str = tokenizer.decode(
#               [input_ids[letter_pos]], ...              # ← index space mismatch
#           )
#
# FIXED code (drop-in replacement):
#
#   if position in ["letter", "letter+1", "letter-1"]:
#       letter_pos, probed_token_str = find_letter_token_position_in_full_text(
#           full_text, tokenizer                          # ← searches full text
#       )                                                 # ← returns full-text index
#       if letter_pos is not None:
#           if position == "letter+1":
#               letter_pos += 1
#           elif position == "letter-1":
#               letter_pos -= 1
#           probed_token_str = tokenizer.decode(
#               [input_ids[letter_pos]], skip_special_tokens=False
#           )
#           position_to_token_indices[position] = [letter_pos]
#           if len(probed_tokens_by_position[position]) == text_idx:
#               probed_tokens_by_position[position].append(probed_token_str)
#       else:
#           # genuine fallback (rare)
#           fallback_positions = get_probe_positions(input_ids, special_token_ids, "last")
#           position_to_token_indices[position] = fallback_positions
#           if len(probed_tokens_by_position[position]) == text_idx:
#               probed_tokens_by_position[position].append(
#                   tokenizer.decode([input_ids[fallback_positions[0]]], skip_special_tokens=False)
#               )
#
# NOTE: The 'completions' argument to extract_all_layers_all_positions_multi_gpu
# is still needed for yes_no / yes_no-1 positions. It is no longer needed for
# letter / letter+1 / letter-1 positions after this fix.
# ---------------------------------------------------------------------------
