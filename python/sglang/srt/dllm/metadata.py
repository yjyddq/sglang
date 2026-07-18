"""Validation helpers for diffusion-LLM token-aligned metadata."""

from operator import index as operator_index
from typing import List


def normalize_step_map_chunk(
    step_map, expected_length: int, rid: str
) -> List[int]:
    """Convert a model-produced step map to validated host integers."""
    if hasattr(step_map, "cpu"):
        step_map = step_map.cpu().tolist()
    else:
        step_map = list(step_map)

    if len(step_map) != expected_length:
        raise RuntimeError(
            f"dLLM step_maps/output_ids mismatch for request {rid}: got "
            f"{len(step_map)} steps and {expected_length} tokens"
        )

    normalized = []
    for value in step_map:
        if isinstance(value, bool):
            raise RuntimeError(
                f"dLLM step map for request {rid} contains a boolean value"
            )
        try:
            value = operator_index(value)
        except TypeError as exc:
            raise RuntimeError(
                f"dLLM step map for request {rid} contains a non-integer value: "
                f"{value!r}"
            ) from exc
        if value <= 0:
            raise RuntimeError(
                f"dLLM step map for request {rid} must use positive 1-based "
                f"steps, got {value}"
            )
        normalized.append(value)
    return normalized
