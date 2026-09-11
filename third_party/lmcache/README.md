# LMCache runtime pin

`VERSION` pins the stock LMCache package installed by `scripts/setup_vllm_tt_plugin.sh`. The
matching `lmcache-t3knic` connector source is part of the canonical TT-Metal target tree.

Stock LMCache is installed as an external dependency and is not vendored in this source repository.
It remains subject to the upstream
[Apache-2.0 license](https://github.com/LMCache/LMCache/blob/v0.4.4/LICENSE). The Moreh-authored
`lmcache-t3knic` connector is provided under the repository-level Apache-2.0 license.
