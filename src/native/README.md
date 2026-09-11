# Native Runtime Ownership

Sources under `src/native/ttnn` are private implementation inputs owned and built by this repository. They are not installed as public headers or library sources.

The client lifecycle sources own Client construction, Fabric session establishment, connection and quarantine state, bootstrap resources, and the single ESTABLISH handshake. They call the versioned opaque `ClientOperations` ABI from the matching installed `TTNN::TTNN` package for request program construction and execution. TTNN retains that operation implementation because it must be built with TTNN and its device kernels; it does not expose a product Client API.

The server lifecycle, tiering, shared-memory mapping, and everything under `server_runtime/` are compiled directly into `libkvcache_manager`. Together they own persistent program creation, device and pinned-memory resources, raw transfers, host backing, tiering, policy snapshots, profile accumulation, and shutdown. TTNN supplies only the protocol/configuration headers and runtime kernel assets used to construct those programs; it does not supply a Server runtime ABI or product Server API.

The build links only to an installed `TTNN::TTNN` from the exact paired tt-metal revision. It must not add a tt-metal source directory to include paths, inject manager sources into TTNN, or recover deleted TTNN Client/Server sources through the patch bootstrap.
