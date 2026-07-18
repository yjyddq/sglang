#!/usr/bin/env python3
"""Exercise dLLM step maps through batch, multi-block, and stream APIs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from sglang.srt.entrypoints.engine import Engine


BLOCK_SIZE = 4
DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/yangjingyi/DARE/models/SDAR-8B-Chat"
ALGORITHM_CONFIG = str(Path(__file__).with_name("low_confidence_step_maps.yaml"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def validate_step_map(
    output_ids: list[int], step_map: list[int], prompt_length: int, context: str
) -> None:
    if len(step_map) != len(output_ids):
        raise RuntimeError(
            f"{context}: got {len(step_map)} steps for {len(output_ids)} tokens"
        )
    if any(
        isinstance(step, bool)
        or not isinstance(step, int)
        or not 1 <= step <= BLOCK_SIZE
        for step in step_map
    ):
        raise RuntimeError(f"{context}: invalid 1-based step map {step_map}")

    first_block_tokens = (-prompt_length) % BLOCK_SIZE
    offset = 0
    if first_block_tokens:
        first_chunk = step_map[:first_block_tokens]
        if sorted(first_chunk) != list(range(1, first_block_tokens + 1)):
            raise RuntimeError(
                f"{context}: invalid prompt-boundary block map {first_chunk}"
            )
        offset = first_block_tokens

    while offset < len(step_map):
        chunk = step_map[offset : offset + BLOCK_SIZE]
        if len(chunk) != BLOCK_SIZE or sorted(chunk) != list(
            range(1, BLOCK_SIZE + 1)
        ):
            raise RuntimeError(
                f"{context}: invalid complete block at output offset {offset}: "
                f"{chunk}"
            )
        offset += BLOCK_SIZE


def get_payload(output: dict, context: str) -> tuple[list[int], list[int]]:
    output_ids = output.get("output_ids", output.get("token_ids"))
    if output_ids is None:
        raise RuntimeError(f"{context}: response has no output token IDs")
    meta_info = output.get("meta_info", {})
    if "step_maps" not in meta_info:
        raise RuntimeError(f"{context}: response is missing meta_info['step_maps']")
    return output_ids, meta_info["step_maps"]


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True
    )
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": "What is 1 + 1? Answer briefly."}],
        tokenize=True,
        add_generation_prompt=True,
    )
    # End on a global dLLM block boundary while still crossing SGLang's
    # 50-token forced-output boundary.
    max_new_tokens = BLOCK_SIZE * 16 - (len(input_ids) % BLOCK_SIZE)
    if max_new_tokens < 50:
        raise RuntimeError(f"Test response is unexpectedly short: {max_new_tokens}")
    if (len(input_ids) + max_new_tokens) % BLOCK_SIZE:
        raise RuntimeError("Test response does not end on a global block boundary")

    sampling_params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": max_new_tokens,
        "ignore_eos": True,
    }
    engine = Engine(
        model_path=args.model_path,
        dtype="bfloat16",
        mem_fraction_static=0.5,
        tp_size=1,
        trust_remote_code=True,
        max_running_requests=2,
        dllm_algorithm="LowConfidence",
        dllm_algorithm_config=ALGORITHM_CONFIG,
        attention_backend="flashinfer",
        stream_output=True,
    )
    try:
        batch_outputs = engine.generate(
            input_ids=[input_ids, input_ids],
            sampling_params=sampling_params,
            return_step_maps=[False, True],
        )
        stream_outputs = list(
            engine.generate(
                input_ids=input_ids,
                sampling_params=sampling_params,
                return_step_maps=True,
                stream=True,
            )
        )
    finally:
        engine.shutdown()

    if not isinstance(batch_outputs, list) or len(batch_outputs) != 2:
        raise RuntimeError(f"Expected two batch responses, got {batch_outputs!r}")
    if "step_maps" in batch_outputs[0].get("meta_info", {}):
        raise RuntimeError("batch request 0 returned unrequested step maps")
    batch_ids, batch_steps = get_payload(batch_outputs[1], "batch request 1")
    validate_step_map(batch_ids, batch_steps, len(input_ids), "batch request 1")

    stream_ids = []
    stream_steps = []
    nonempty_chunks = 0
    for chunk_index, output in enumerate(stream_outputs):
        chunk_ids = output.get("output_ids", output.get("token_ids")) or []
        if not chunk_ids:
            continue
        ids, steps = get_payload(output, f"stream chunk {chunk_index}")
        if len(ids) != len(steps):
            raise RuntimeError(
                f"stream chunk {chunk_index}: got {len(ids)} tokens and "
                f"{len(steps)} steps"
            )
        stream_ids.extend(ids)
        stream_steps.extend(steps)
        nonempty_chunks += 1

    if nonempty_chunks < 2:
        raise RuntimeError(
            f"Expected incremental streaming across multiple chunks, got "
            f"{nonempty_chunks}"
        )
    validate_step_map(stream_ids, stream_steps, len(input_ids), "stream response")
    if len(stream_ids) != max_new_tokens:
        raise RuntimeError(
            f"Stream returned {len(stream_ids)} tokens, expected {max_new_tokens}"
        )

    result = {
        "passed": True,
        "prompt_length": len(input_ids),
        "output_length": max_new_tokens,
        "batch_step_map_length": len(batch_steps),
        "stream_step_map_length": len(stream_steps),
        "stream_nonempty_chunks": nonempty_chunks,
        "batch_step_map_prefix": batch_steps[:12],
        "stream_step_map_prefix": stream_steps[:12],
    }
    print("DLLM_STEP_MAP_ENGINE_RESULT=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
