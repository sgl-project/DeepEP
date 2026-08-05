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
- the GDRCopy user-space library (`libgdrapi.so`);
- the loaded GDRCopy kernel module and `/dev/gdrdrv` passed into the container;
- the RDMA/NVLink environment required by the selected DeepEP transport.

At import time, the package checks the supported platform, PyTorch and CUDA
versions, CUDA driver/device availability, `libgdrapi`, and `/dev/gdrdrv`.
Failures raise an actionable `ImportError`. Transport-specific RDMA and NVLink
topology cannot be proven at import time and is validated when DeepEP
initializes the selected transport.
