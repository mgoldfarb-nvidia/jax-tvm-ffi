# CuTeDSL + JAX Softmax Example

High-performance softmax kernel written in [CuTeDSL](https://github.com/NVIDIA/cutlass), integrated with JAX via `jax-tvm-ffi`.

## Features

- **CuTeDSL kernels**: Forward and backward passes using online softmax algorithm
- **JIT compilation**: Works with `@jax.jit`
- **Autodiff**: Full gradient support via `jax.custom_vjp`
- **Multi-GPU**: Compatible with `jax.shard_map`

## Installation

```bash
pip install jax-tvm-ffi[cutedsl]
```

Or install dependencies separately:
```bash
pip install jax-tvm-ffi nvidia-cutlass-dsl
```

## Quick Start

```bash
python -m examples.cutedsl.jax_softmax
```

To embed the compiled native objects in StableHLO instead of registering live
TVM FFI function handles, install the ORCJIT 0.1.1+ extra and run the serialized version:

```bash
pip install "jax-tvm-ffi[cutedsl,orcjit]"
python -m examples.cutedsl.jax_softmax_serialized
```

The serialized path currently also requires an unreleased CuTe DSL compiler
extension exposing `CuteCompiler.set_tvm_ffi_self_initialize_cuda`; public
`nvidia-cutlass-dsl` 4.6 does not contain it. Use a coordinated DKG source build
until that extension and ORCJIT 0.1.1 are published.

The serialized example uses the CuTe DSL artifact compiler to produce a
`SerializedFunction` containing the TVM FFI object bytes, exported function
name, and SHA-256 digest. It embeds the bytes and name in the HLO without writing
an object file. The installed ORCJIT extension must support loading object bytes
through its existing execution-session API. JAX TVM FFI caches the loaded module
weakly by object SHA-256 and resolves each requested function from it. Live JAX
executables own the shared module. The reusable compiler entry point is
`jax_tvm_ffi.cutlass.compile_to_object`; it currently relies on CuTe DSL's
experimental fine-grained compilation API.
Pass a `compile_options` key/value mapping to override lowering options accepted
by `CuteCompiler.add_compile_option`; the TVM FFI ABI remains mandatory.
The generated wrapper owns lazy CUDA initialization and unloading. The helper
loads the installed CuTe runtime libraries with process-global visibility so
ORCJIT can resolve the object's runtime symbols; call
`jax_tvm_ffi.cutlass.load_runtime()` when loading a previously serialized JAX
executable in a fresh process without recompiling the kernel first.
Identical artifacts are reused from a process-local compiled-object cache. Pass
`no_cache=True` to force recompilation, including when compiler options are used
for diagnostic output or other side effects.

## Usage

```python
from examples.cutedsl.jax_softmax import softmax, register_softmax_ops
import jax
import jax.numpy as jnp

# Register kernels for your configuration
register_softmax_ops(N=1024, dtype=jnp.bfloat16)

# Create input
x = jax.random.normal(jax.random.key(0), (32, 1024), dtype=jnp.bfloat16)

# Use with JIT
y = jax.jit(softmax)(x)

# Use with grad
dx = jax.grad(lambda x: softmax(x).sum())(x)
```

### Multi-GPU with shard_map

```python
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

devices = jax.devices("gpu")
if len(devices) >= 2:
    mesh = Mesh(np.array(devices[:2]), axis_names=('batch',))
    x_sharded = jax.device_put(x, NamedSharding(mesh, P('batch', None)))
    y_sharded = jax.shard_map(
        softmax, mesh=mesh,
        in_specs=(P('batch', None),), out_specs=P('batch', None),
    )(x_sharded)
```

## Files

| File | Description |
|------|-------------|
| `softmax.py` | CuTeDSL kernel implementations (forward + backward) |
| `jax_softmax.py` | JAX integration of the softmax kernel |
| `jax_softmax_serialized.py` | JAX integration with object bytes embedded in StableHLO |
