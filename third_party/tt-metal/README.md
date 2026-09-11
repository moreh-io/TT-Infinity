# TT-Metal patch bundle

This directory is the canonical TT-Metal patch bundle for kvcache-manager.

- `REVISION` identifies the exact clean upstream checkout.
- `patches/series` and `patches/SHA256SUMS` define the ordered, immutable patch payloads.
- `TARGET_TREE` identifies the complete prepared Git index.
- `LICENSE`, `NOTICE`, and `LICENSE_understanding.txt` are copied from the pinned upstream revision.

The series is organized by reviewable functionality rather than a fixed patch count:

1. static inter-mesh Fabric;
2. cross-node link diagnostics and KV data-path operations;
3. the KVC protocol and device runtime;
4. the Qwen3 model runtime; and
5. the LMCache T3KNIC connector.

Apply it with `scripts/prepare_tt_metal.sh`. Maintainers can regenerate it from a staged or committed
TT-Metal source tree with:

```bash
scripts/generate_tt_metal_patches.sh --tt-metal-dir /path/to/tt-metal
```

The generator rejects every changed source path that has no functional owner and verifies the
complete generated target tree before replacing the manifest.

## License and Attribution

Upstream-derived portions remain subject to the upstream Apache-2.0 license and copyright. Moreh
licenses its modifications under Apache-2.0. The upstream legal files included in this directory
apply to the pinned base revision and are preserved when the patch bundle is regenerated.
