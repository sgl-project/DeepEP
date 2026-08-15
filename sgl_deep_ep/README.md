# sgl-deep-ep

`sgl-deep-ep` is SGLang's binary distribution of
[DeepEP](https://github.com/sgl-project/DeepEP). The distribution name is
`sgl-deep-ep`, while the Python import remains compatible with DeepEP:

```python
import deep_ep
```

## Installation

The PyPI package targets CUDA 13 and installs with:

```bash
pip install sgl-deep-ep
```

CUDA 12.9 wheels are published separately in the
[SGLang wheel repository](https://github.com/sgl-project/whl). Install the
matching PyTorch CUDA 12.9 build first, then install the wheel from the `cu129`
index or release URL.

Each wheel is specific to a CPython version, CPU architecture, and CUDA major.
The package checks these constraints when `deep_ep` is imported.

## Host prerequisites

The wheel contains DeepEP's Python code and CUDA extension. It does not and
cannot provision the host kernel or RDMA stack. Every host must provide:

- a compatible NVIDIA driver and CUDA runtime;
- the RDMA/NVLink environment required by the selected DeepEP transport.

At import time, the package checks the supported platform, PyTorch and CUDA
versions, and CUDA driver/device availability. Failures raise an actionable
`ImportError`. Transport-specific prerequisites are validated when DeepEP
initializes the selected transport.

CUDA 13 wheels use DeepEP v2 and its NCCL Gin transport. Release and runtime
images pin `nvidia-nccl-cu13==2.30.7`; this path does not require GDRCopy or a
`/dev/gdrdrv` device. NVSHMEM remains a build and runtime dependency because the
current v2 extension still links the legacy DeepEP methods.

CUDA 12.9 wheels use the legacy NVSHMEM transport. Their low-latency and
internode paths require an IBGDA-capable host, provided either by NVIDIA driver
configuration or by GDRCopy (`libgdrapi.so` plus a usable `/dev/gdrdrv`). See the
[NVSHMEM setup guide](https://github.com/sgl-project/DeepEP/blob/sgl-deepep-packaging/docs/nvshmem.md)
for both supported configurations.

## Wheel release workflow

Wheels are built and published by SGLang's
[Release sgl-deep-ep workflow](https://github.com/sgl-project/sglang/actions/workflows/release-whl-deepep.yml).
Before starting a release, merge the required implementation changes into the
matching implementation branch and merge the packaging overlay into
`sgl-deepep-packaging`:

- CUDA 13, x86_64 and aarch64: `sgl-deepep` (DeepEP v2);
- CUDA 12.9, x86_64: `sgl-deepep-cu12-x86`;
- CUDA 12.9, aarch64: `sgl-deepep-cu12-arm`.

Run the workflow manually with:

- `version`: the public package version, without a leading `v`;
- `target`: `cu129`, `cu130`, or `all`;
- `packaging-ref`: the packaging overlay branch, tag, or commit (normally
  `sgl-deepep-packaging`).

The CUDA 12.9 matrix builds CPython 3.10 and 3.12 wheels. The CUDA 13.0 matrix
builds CPython 3.10, 3.11, 3.12, and 3.13 wheels. Both matrices build for
`x86_64` and `aarch64`.

For each selected CUDA target, the workflow uploads CUDA-tagged wheels to the
corresponding `cu129` or `cu130` index in
[sgl-project/whl](https://github.com/sgl-project/whl). CUDA 13.0 wheels are also
published to PyPI as `sgl-deep-ep`, without a CUDA suffix in the package name.
Publishing requires the `GH_PAT_FOR_WHL_RELEASE` and
`SGL_DEEP_EP_PYPI_TOKEN` repository secrets.

After the workflow completes, check that every expected Python and architecture
artifact was published, install the appropriate wheel in a clean environment,
and verify both `import deep_ep` and a multi-GPU DeepEP or SGLang workload.
