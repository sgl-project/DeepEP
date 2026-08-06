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

Low-latency and internode transports require an IBGDA-capable host. This can be
provided either by the NVIDIA driver configuration or by GDRCopy
(`libgdrapi.so` plus a usable `/dev/gdrdrv`). See the
[NVSHMEM setup guide](../docs/nvshmem.md) for both supported configurations.
