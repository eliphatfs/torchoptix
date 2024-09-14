# TorchOptiX

Modular wrapper for using OptiX with PyTorch.

## Requirements

Most requirements are the same as running OptiX.

+ **Hardware:** All NVIDIA GPUs of Compute Capability 5.0 (Maxwell) or higher are supported.
+ **Driver:** An driver version of R515+ is required. You may check with `nvidia-smi`.
+ **Python:** 3.8 or higher.
+ **PyTorch:** 2 or higher is recommended. May also work for older versions.

### Running in containers like Docker

To run inside a container, you need to configure the driver for OptiX. You can choose from the following options:

1. Set `ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics` and `ENV PYOPENGL_PLATFORM egl` in `Dockerfile` when building the image.
2. Set `-e NVIDIA_DRIVER_CAPABILITIES=graphics,compute,utility` when creating the container.
3. Copy or mount `/usr/lib/x86_64-linux-gnu/libnvoptix.so.<version>` on the host or download a same version of the library to `/usr/lib/x86_64-linux-gnu/libnvoptix.so.1` in the container; copy `/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.<version>` on the host or download a same version of the library to `/usr/lib/x86_64-linux-gnu/libnvidia-rtcore.so.<version>` in the container.

## Installation

Stable release (Windows or Linux 64-bit):

```bash
pip install torchoptix
```

Development (or if you are not using a common system supported by prebuilt binaries):

```bash
pip install git+https://github.com/eliphatfs/torchoptix.git
```

You will need to have `CUDA_HOME` set to compile or develop. The code does not depend on CUDA runtime libraries or `nvcc`, but needs CUDA driver API header and link libraries.

To regenerate resources for device code if you modified it in development:

```
bash generate.sh
```

## Usage

### Example Wrapper

```python
import torch
from typing import Tuple

class TorchOptiX:
    @torch.no_grad()
    def __init__(self, verts: torch.Tensor, tris: torch.IntTensor) -> None:
        self.handle = None
        import torchoptix
        self.optix = torchoptix
        self.verts = verts.contiguous()
        self.tris = tris.contiguous()
        self.handle = self.optix.build(
            self.verts.data_ptr(), self.tris.data_ptr(),
            len(verts), len(tris)
        )

    @torch.no_grad()
    def query(self, rays_o: torch.Tensor, rays_d: torch.Tensor, far: float) -> Tuple[torch.Tensor]:
        # out_i starts at 0 and is 0 when not hit.
        # you can decide hits via `t < far`.
        out_t = rays_o.new_empty([len(rays_o)])
        out_i = rays_o.new_empty([len(rays_o)], dtype=torch.int32)
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()
        self.optix.trace_rays(
            self.handle,
            rays_o.data_ptr(),
            rays_d.data_ptr(),
            out_t.data_ptr(), out_i.data_ptr(),
            far, len(rays_o)
        )
        return out_t, out_i

    def __del__(self):
        if self.handle is not None and self.optix is not None and self.optix.release is not None:
            self.optix.release(self.handle)
            self.handle = None
```

Example:

```python
import torch.nn.functional as F
accel = TorchOptiX(torch.randn(10, 3).cuda(), torch.randint(0, 10, [5, 3]).cuda().int())
t, i = accel.query(torch.randn(20, 3).cuda(), F.normalize(torch.randn(20, 3).cuda(), dim=-1), far=32767)
print(t, i, sep='\n')
```

### Low-level API

```
NAME
    torchoptix - Modular OptiX ray tracing functions interop with PyTorch.

FUNCTIONS
    build(...)
        build(verts, tris, n_verts, n_tris) -> handle

        Build OptiX acceleration structure.

    release(...)
        release(handle)

        Release OptiX acceleration structure.

    set_log_level(...)
        set_log_level(level)

        Set OptiX log level (0-4).

    trace_rays(...)
        trace_rays(handle, rays_o, rays_d, out_t, out_i, t_max, n_rays)

        Trace rays with OptiX.
```

`verts`, `tris`, `rays_o`, `rays_d`, `out_t`, `out_i` are CUDA device pointers.
`tris` and `out_i` are contiguous `int32` arrays, and others are `float32` arrays.
`t_max` (float) is maximum distance of ray to trace.

The functions need to be called when the same CUDA device context is active.
The APIs are not thread-safe on the same device.
In PyTorch, to run on multiple devices you need to use distributed parallelism, and each process runs a device. Multi-threading on devices is not supported.

It is not necessary that the arrays originate from PyTorch. It can be allocated with native CUDA.

## Citation

```bibtex
@misc{TorchOptiX,
  title = {TorchOptiX},
  howpublished = {\url{https://github.com/eliphatfs/torchoptix}},
  note = {Accessed: 2024-09-13}
}
```
