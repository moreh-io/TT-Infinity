# SemiAnalysisAI AIPerf patch bundle

This bundle pins the AIPerf fork used by InferenceX and adds offline loading from an absolute local
tokenizer directory. `scripts/prepare_aiperf.sh` verifies the base revision, patch checksum, and
resulting Git tree before changing a checkout.

The patch does not change the InferenceX scenario, Weka replay semantics, or result calculation.
It only resolves an existing local tokenizer path before the Hugging Face offline-cache lookup.

## License and Attribution

Upstream-derived portions remain subject to the upstream Apache-2.0 license and copyright. Moreh
licenses its modifications under Apache-2.0. `LICENSE` and `ATTRIBUTIONS.md` are copied from the
pinned upstream revision; the assets described by `ATTRIBUTIONS.md` are not included in this patch
bundle.
