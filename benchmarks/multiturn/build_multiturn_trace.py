#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Moreh
#
# SPDX-License-Identifier: Apache-2.0

"""Create one deterministic, sequential multi-turn Weka trace for TT tests."""

from __future__ import annotations

import argparse
from pathlib import Path

import orjson

from aiperf.dataset.loader.weka_trace_models import WekaTrace


def _request(timestamp: float, input_tokens: int, output_tokens: int) -> dict:
    block_size = 64
    if input_tokens % block_size:
        raise ValueError("input token counts must be multiples of 64")
    return {
        "t": timestamp,
        "type": "s",
        "model": "qwen3-32b-tt-multiturn",
        "in": input_tokens,
        "out": output_tokens,
        "hash_ids": list(range(input_tokens // block_size)),
        "input_types": ["text"],
        "output_types": ["text"],
        "stop": "end_turn",
        # Chain detection requires the previous recorded request to have ended.
        # AIPerf still waits for the real response before issuing the next turn.
        "api_time": 0.001,
        "think_time": 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--turns", type=int, default=5)
    parser.add_argument("--first-turn-tokens", type=int, default=4096)
    parser.add_argument("--tokens-per-turn", type=int, default=768)
    parser.add_argument("--inter-turn-delay", type=float, default=0.01)
    parser.add_argument("--output-tokens", type=int, default=1)
    args = parser.parse_args()

    if args.turns < 2:
        parser.error("at least two turns are required")
    if args.first_turn_tokens <= 0 or args.tokens_per_turn <= 0:
        parser.error("token lengths must be positive")
    if args.inter_turn_delay <= 0.001:
        parser.error("--inter-turn-delay must exceed the synthetic 0.001s API time")
    if args.output_tokens <= 0:
        parser.error("output tokens must be positive")

    trace = WekaTrace.model_validate(
        {
            "id": "qwen3-32b-tt-prefix-reuse-5turn-4k-7k",
            "models": ["qwen3-32b-tt-multiturn"],
            "block_size": 64,
            "hash_id_scope": "local",
            "tool_tokens": 0,
            "system_tokens": 0,
            "requests": [
                _request(
                    turn * args.inter_turn_delay,
                    args.first_turn_tokens + turn * args.tokens_per_turn,
                    args.output_tokens,
                )
                for turn in range(args.turns)
            ],
            "totals": None,
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stale in args.output_dir.glob("000-qwen3-32b-tt-prefix-reuse-*.json"):
        stale.unlink()
    output = args.output_dir / "000-qwen3-32b-tt-prefix-reuse-5turn-4k-7k.json"
    output.write_bytes(orjson.dumps(trace.model_dump(by_alias=True)) + b"\n")
    print(f"wrote multi-turn prefix-reuse trace with {args.turns} turns to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
