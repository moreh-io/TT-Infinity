# Third-Party Licenses and Notices

This document distinguishes third-party material distributed by this source repository from
packages installed separately at runtime. Copyright in upstream material remains with the
respective copyright holders. Moreh-authored modifications are provided under the Apache License,
Version 2.0, as stated in the repository-level [LICENSE](LICENSE).

## Distributed Patch Bundles

### TT-Metal

- Upstream: <https://github.com/tenstorrent/tt-metal>
- Base revision: `ab49094f2c46bb6e4cc3b23fb231980db3376207`
- License: Apache-2.0

The files under `third_party/tt-metal/patches` contain modifications to TT-Metal. Copies of the
upstream [`LICENSE`](third_party/tt-metal/LICENSE), [`NOTICE`](third_party/tt-metal/NOTICE), and
[`LICENSE_understanding.txt`](third_party/tt-metal/LICENSE_understanding.txt) from the pinned base
revision accompany the patch bundle.

### vLLM TT Plugin

- Upstream: <https://github.com/tenstorrent/vllm-tt-plugin>
- Base revision: `3066670825b314d164e2159fc383b0b08ee0784b`
- License: Apache-2.0

The files under `third_party/vllm-tt-plugin/patches` contain modifications to the vLLM TT Plugin.
Copies of the upstream [`LICENSE`](third_party/vllm-tt-plugin/LICENSE),
[`NOTICE`](third_party/vllm-tt-plugin/NOTICE), and
[`LICENSE_understanding.txt`](third_party/vllm-tt-plugin/LICENSE_understanding.txt) from the pinned
base revision accompany the patch bundle.

### SemiAnalysisAI AIPerf

- Upstream: <https://github.com/SemiAnalysisAI/aiperf>
- Base revision: `754356e9a39acc6cc6afb242d123bb57c3fb6f75`
- License: Apache-2.0

The files under `third_party/aiperf/patches` contain modifications to AIPerf. Copies of the
upstream [`LICENSE`](third_party/aiperf/LICENSE) and
[`ATTRIBUTIONS.md`](third_party/aiperf/ATTRIBUTIONS.md) from the pinned base revision accompany the
patch bundle. The assets described by the upstream attribution file are not vendored in this
source repository.

## External Runtime Dependencies

The following packages are installed into external environments and their source is not vendored
in this repository:

| Package | Pinned version | License |
| --- | --- | --- |
| [vLLM](https://github.com/vllm-project/vllm) | 0.24.0 | [Apache-2.0](https://github.com/vllm-project/vllm/blob/v0.24.0/LICENSE) |
| [LMCache](https://github.com/LMCache/LMCache) | 0.4.4 | [Apache-2.0](https://github.com/LMCache/LMCache/blob/v0.4.4/LICENSE) |

These dependencies remain subject to their own license terms. A wheel, container, appliance, or
other artifact that bundles them must include the license and attribution material applicable to
the contents of that artifact.
