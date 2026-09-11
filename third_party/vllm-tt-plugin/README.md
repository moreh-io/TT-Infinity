# vLLM TT plugin patch bundle

This directory is the canonical patch bundle for the vLLM TT plugin used by the Qwen3 cross-node
KV example.

- `REPOSITORY` and `REVISION` identify the exact clean upstream checkout.
- `patches/series` and `patches/SHA256SUMS` define the ordered, immutable patch payloads.
- `TARGET_TREE` identifies the complete prepared Git index.
- `VLLM_VERSION` pins the matching stock vLLM package.
- `LICENSE`, `NOTICE`, and `LICENSE_understanding.txt` are copied from the pinned upstream revision.

`0001-qwen3-cross-node-kv.patch` is one integration unit because its scheduler, model runner, worker,
and offline example changes share one request lifecycle contract. Use
`scripts/prepare_vllm_tt_plugin.sh` instead of applying it manually.

## License and Attribution

Upstream-derived portions remain subject to the upstream Apache-2.0 license and copyright. Moreh
licenses its modifications under Apache-2.0. Stock vLLM is installed separately and remains subject
to its own license terms.
