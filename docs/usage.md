# Using TT-Infinity with vLLM and LMCache

TT-Infinity uses its `kvcache-manager` runtime to store and restore vLLM KV cache from Galaxy to a T3K `Server` through LMCacheConnector. When vLLM loads `T3knicKVConnector`, the Galaxy `Client` accesses T3K over TT-Fabric.

## Integration Flow

See [Architecture](architecture.md) for prefix lookup and KV block save/restore data flow. This document covers deployment setup and execution.

## Environment Setup

Follow the [README Quick Start](../README.md#quick-start) to prepare TT-Infinity and tt-metal on Galaxy and T3K and the vLLM TT plugin environment on Galaxy. Both hosts must use the same tt-metal base revision and patch series and must be connected through a supported intermesh configuration.

Place the Hugging Face `Qwen/Qwen3-32B` checkpoint at a local path on Galaxy. With the `huggingface_hub` CLI:

```bash
hf download Qwen/Qwen3-32B --local-dir /path/to/Qwen3-32B
```

Set `MODEL` for `run_qwen3_32b.sh` to this directory.

### Intermesh Endpoint Configuration

After applying the tt-metal patches, edit the following tables in `lmcache-t3knic/lmcache_t3knic/t3knic_lmcache/intermesh.py` on both hosts to match the physical connection:

```python
CABLE_ENDPOINT_CHIPS = {
    0: (28, 0),
    1: (24, 1),
    2: (20, 3),
    3: (16, 2),
}

CABLE_ENDPOINT_BDFS = {
    1: {
        "galaxy": "0000:c1:00.0",
        "t3k": "0000:4b:00.0",
    },
}
```

Each `CABLE_ENDPOINT_CHIPS` value is `(Galaxy physical chip ID, T3K physical chip ID)`. `CABLE_ENDPOINT_BDFS` gives both PCI BDFs for the same cable index. The run scripts use `LINK=1`; if another index is required, change `LINK=1` in both `run_qwen3_32b.sh` and `run_multiturn_prefix_reuse.sh`. Ethernet channels are discovered from live topology at runtime.

## LMCache Configuration

The default Quick Start configuration is in `configs/lmcache_qwen3_32b.yaml`.

```yaml
chunk_size: 128
local_cpu: false
max_local_cpu_size: 128.0
remote_serde: "naive"
save_unfull_chunk: true
save_decode_cache: false
lookup_timeout_ms: 60000
```

`chunk_size` must equal vLLM `block_size`; the reference configuration uses `128` for both. For direct TT-device KV transfer, disable the local CPU cache, set `remote_serde` to `naive`, and do not set `remote_url`. Other serialization backends or an intermediate local cache may copy tensors and invalidate the connector side channel that identifies device KV.

These values demonstrate the connector contract, not a complete deployment configuration. Select model shape, KV-cache capacity, and timeouts for the workload.

## Qwen3-32B Execution

`run_qwen3_32b.sh` validates the deployment values exported during Quick Start and invokes tt-metal's `run_cross_node_ttft.sh`.

| Variable | Description |
| --- | --- |
| `TT_METAL_ROOT` | Patched and built tt-metal on Galaxy |
| `PLUGIN` | vLLM TT plugin checkout prepared by Quick Start |
| `PLUGIN_VENV` | Plugin Python environment prepared by Quick Start |
| `T3K_HOST` | T3K host reachable through non-interactive SSH |
| `T3K_REPO` | T3K tt-metal path; optional when identical to the Galaxy path |
| `MODEL` | Qwen3-32B checkpoint on Galaxy |
| `LMCACHE_CONFIG_FILE` | LMCache YAML; defaults to `configs/lmcache_qwen3_32b.yaml` |

Defaults use dual link, eight server shards, and per-chip wire mode. Override `T3KNIC_INTERMESH_LINK_COUNT`, `T3KNIC_SERVER_SHARDS`, `T3KNIC_WIRE_MODE`, `REQUESTS`, `PROMPT_TOKENS`, `MAX_MODEL_LEN`, `OSL`, or `OUT` as needed.

```bash
./run_qwen3_32b.sh
```

If a required value or local tt-metal, plugin, Python, model, or LMCache configuration is missing, the script exits with an error and points to this document. After validation, the runner starts the T3K server over SSH, starts Galaxy vLLM, runs the Cold/Warm workload, and terminates both processes. vLLM prefix caching is disabled and the resumed-prefill trace is prepared before measurement.

The first run of a new model/cache configuration may take several minutes while weights are converted to the TT tensor cache. Progress is printed and the resulting cache is reused. The first workload may also include JIT-program and trace preparation, so use it only to prepare the system and rerun the complete script for steady-state results. Changing model shape, ISL/OSL, or cache configuration may require new programs or traces.

## AIPerf Multi-Turn Prefix-Reuse Execution

`run_multiturn_prefix_reuse.sh` keeps the same cross-node lifecycle and uses SemiAnalysisAI AIPerf's `inferencex-agentx-mvp` scenario as its trace-replay driver. That scenario name is an external AIPerf interface; this project supplies its own synthetic five-turn Weka trace from `benchmarks/multiturn/datasets` and does not run the InferenceX workload. Nominal ISLs are 4096, 4864, 5632, 6400, and 7168 with OSL 1. The chat template may make server-reported ISLs slightly larger.

Prepare AIPerf on Galaxy using the [README AIPerf example](../README.md#aiperf-multi-turn-prefix-reuse-example). Installation, workload execution, and result storage all occur on Galaxy. T3K needs only tt-metal and `kvcache-manager`. In addition to the Qwen variables, set:

| Variable | Description |
| --- | --- |
| `AIPERF_VENV` | Separate Python 3.11 environment containing patched AIPerf |
| `HF_HOME` | Writable offline Hugging Face cache root; tokenizer files are read from local `MODEL` |
| `OUT_MULTITURN` | New result directory; execution is rejected if it already contains results |
| `MULTITURN_INPUT_DIR` | Weka trace directory; defaults to `benchmarks/multiturn/datasets` |

Run KVM ON with the command below. `KVM_DISABLE` defaults to 0. In environments that require device reservation, wrap the entire command with the deployment's reservation mechanism.

```bash
./run_multiturn_prefix_reuse.sh
```

Defaults are `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=1`, `CONC=1`, `MULTITURN_REQUEST_COUNT=5`, and a 90-second timeout. `CONC=1` runs one conversation trajectory at a time; dependent turns are sent after the previous response completes.

Before measurement, the runner stores a separate 4096-token synthetic prefix and restores it with a 1024-token tail. These two prewarm requests are excluded from AIPerf results and complete compilation and trace capture for the batch-1 resumed-prefill path outside measurement. Results therefore represent steady-state TTFT. Set `PREWARM_RESUMED_PREFILL=0` to disable this behavior.

For the KVM OFF baseline, preserve the model, trace, and AIPerf options, set `KVM_DISABLE=1`, and use a new output directory.

```bash
KVM_DISABLE=1 OUT_MULTITURN=/path/to/new-kvm-off-results ./run_multiturn_prefix_reuse.sh
```

The local trace is deliberately outside the official InferenceX dataset and duration requirements, so the runner uses AIPerf `--unsafe-override`. This development microbenchmark has `submission_valid=false` and is not an InferenceX workload or score. Weka denotes the trace format for timestamps, dependencies, token lengths, and reusable-prefix identity; it does not imply use of Weka storage.

A successful run contains five records with `turn_index` 0–4 in `$OUT_MULTITURN/aiperf_artifacts/profile_export.jsonl`, with no errors or cancellations. Inspect per-turn ISL and TTFT with:

```bash
jq -r '
  ["Turn", "ISL", "TTFT(ms)"],
  [.metadata.turn_index + 1,
   .metrics.input_sequence_length.value,
   .metrics.time_to_first_token.value]
  | @tsv
' "$OUT_MULTITURN/aiperf_artifacts/profile_export.jsonl" | column -t
```

`profile_export_aiperf.csv` contains aggregate average, minimum, maximum, and percentiles rather than per-request records. With OSL 1, this experiment compares post-restore prefill/TTFT, not generation throughput or conversational quality. Validate semantic correctness and KVM ON/OFF output agreement separately.

## Verifying Cache Reuse

The runner sends the same 3,296-token prompt once as a Cold miss and twice as Warm cache-hit requests. It must exit with status zero; `$OUT/vllm.log` and `$OUT/server.log` must show zero hit tokens for Cold and a full 3,296-token hit for both Warm requests. All configured server and client shards must reach ready/ESTABLISH, and the Warm 1 and Warm 2 responses must match each other.

See [README Performance](../README.md#performance) for measurements. On a Warm hit, KV is restored and only the final block tail is forwarded to produce logits. Cold computes the full prompt in one pass, while Warm reads restored KV and computes the tail; numerical differences may therefore change generated text at the tail. A fast request alone does not prove cache correctness.

Validate KV-cache correctness in a separate run by comparing byte digests at the store/load boundary. The reference integration uses `T3KNIC_KVDIFF=1`. A run without an actual GET, with mismatched digests, or with different Warm 1 and Warm 2 responses is not considered a successful cache hit.

## Shutdown Order

The runner closes the T3K server only after every Galaxy client finishes in-flight operations and closes. Mesh and device resources are released after both roles terminate. `Client.close()` and `Server.close()` clean up session and runtime resources; removing stored values requires a separate remove operation.

## Low-level Client/Server API

Integrations other than LMCacheConnector may use `Client` and `Server` directly from Python or C++. The following intranode example uses different devices on one host.

### Python

```python
import ttnn

from kvcache_manager import Client, Server

server = Server(server_submesh)
server_node = server_submesh.get_fabric_node_id(ttnn.MeshCoordinate(0, 0))
client = Client(client_submesh, server_node)

try:
    client.put(key, value_tensor, value_len_bytes)
    client.flush()
    hit = client.get(key, output_tensor)
finally:
    client.close()
    server.close()
```

### C++

```cpp
#include <kvcache_manager/client.hpp>
#include <kvcache_manager/server.hpp>

kvcache_manager::Server server(server_submesh.get());
auto server_node = server_submesh->get_fabric_node_id(
    tt::tt_metal::distributed::MeshCoordinate{0, 0});
kvcache_manager::Client client(client_submesh.get(), server_node);

client.put(key, value_tensor, value_len_bytes);
client.flush();
const bool hit = client.get(key, output_tensor);

client.close();
server.close();
```

The complete C++ API is under [`include/kvcache_manager`](../include/kvcache_manager). Python `Client.delete(key)` corresponds to C++ `Client::remove(key)`.

The caller owns all `MeshDevice` objects. Close every client before the server and release meshes only after both roles are closed. `Client.close()` drains PUTs issued by that client before releasing local session resources. Normal close does not delete stored values.

See [README Quick Start](../README.md#quick-start) for installation.
