# CuTeDSL + JAX Softmax Example

High-performance softmax kernel written in [CuTeDSL](https://github.com/NVIDIA/cutlass), integrated with JAX via `jax-tvm-ffi`.

## Features

- **CuTeDSL kernels**: Forward and backward passes using online softmax algorithm
- **JIT compilation**: Works with `@jax.jit`
- **Autodiff**: Full gradient support via `jax.custom_vjp`
- **Multi-GPU**: Compatible with `jax.shard_map`

## Installation

From the repository root:

```bash
pip install -e ".[cutedsl]"
```

Or install dependencies separately:
```bash
pip install -e . "nvidia-cutlass-dsl>=4.9.0.dev1"
```

## Quick Start

```bash
python -m examples.cutedsl.jax_softmax
```

To embed the compiled native objects in StableHLO instead of registering live
TVM FFI function handles, install the pinned ORCJIT 0.1.3 extra and run the serialized version:

```bash
pip install -e ".[cutedsl,orcjit]"
python -m examples.cutedsl.jax_softmax_serialized
```

CuTe DSL 4.9.0.dev1 or newer can dump a CUDA TVM-FFI compiled handle to object
bytes, which ORCJIT 0.1.3 loads. The object
preserves the kernel's lazy CUDA initialization and exports
`__tvm_ffi_module_init` for explicit initialization.

The serialized example uses `cute.compile[EnableTVMFFI]` and serializes the
compiled handle into a `SerializedFunction` containing the TVM FFI object bytes
and exported function name. It embeds the bytes and name in the HLO without
writing an object file. The installed ORCJIT extension loads object bytes
through its existing execution-session API. Each custom-call instantiation
loads a module and retains it for the executable's lifetime. The
reusable entry point is `jax_tvm_ffi.cutlass.compile_to_object`. Pass additional
compiler flags in the `compile_options` string; they are appended after
the helper's defaults and can override them. The helper always adds
`--enable-tvm-ffi`. CuTe sets up its runtime during compilation. When loading a
previously serialized artifact in a fresh process, explicitly call
`jax_tvm_ffi.cutlass.load_runtime()` to make the runtime symbols available to
ORCJIT.

Call `jax_tvm_ffi.register_builtin_targets(platform="gpu")` during application
setup before compiling or loading payload-backed computations. The call
factories do not register the target. Registration does not initialize a JAX
backend and can run before distributed initialization or backend configuration.

## Export and AOT deployment

The built-in payload APIs support **CUDA GPUs only** (`platform="gpu"`).

The producer compiles a `(32, 1024)` float32 softmax and writes two artifacts.
Both contain the compiled TVM-FFI object bytes and exported function name:

```bash
pip install -e ".[cutedsl,orcjit]" flatbuffers
python -m examples.cutedsl.export_softmax /tmp/softmax-artifacts
```

Only the JAX export format requires `flatbuffers`. To create just the serialized
executable without that dependency, select `--format executable`:

```bash
pip install -e ".[cutedsl,orcjit]"
python -m examples.cutedsl.export_softmax /tmp/softmax-artifacts --format executable
```

Add `--allow-cuda-graph` to the producer command to select the graph-compatible
target. The selected target is preserved in both artifact formats; consumer
commands are the same. JAX export allows the selected target explicitly.

Run each consumer command in a new process:

```bash
# Load the exported JAX program, compile it with XLA, then execute.
python -m examples.cutedsl.run_exported_softmax /tmp/softmax-artifacts/softmax.jax

# Load the AOT executable and execute without recompiling the JAX computation.
python -m examples.cutedsl.run_exported_softmax /tmp/softmax-artifacts/softmax.pkl --format executable
```

The [producer](export_softmax.py) uses `jax.export.export(...).serialize()` for
`softmax.jax` and `jax.jit(...).lower(...).compile()` followed by
`serialize_executable.serialize(...)` for `softmax.pkl`. Export explicitly
allows the predefined `jax_tvm_ffi.payload_call` custom call via
`DisabledSafetyCheck.custom_call`; the bridge and embedded object's compatibility
are the application's responsibility.

The [consumer](run_exported_softmax.py) imports ORCJIT, loads the CuTe runtime,
and registers the predefined payload target. Both producer and consumer register
the target explicitly during setup. The consumer never imports `softmax.py` or
the producer's kernel-construction code. ORCJIT restores the function from the
embedded object, and the consumer checks its result against NumPy. Neither
consumer path recompiles the CuTe kernel. The `jax.export` path still compiles
the surrounding JAX computation with XLA before execution.

Both processes require a CUDA-enabled JAX installation, this branch of
`jax-tvm-ffi`, ORCJIT, and the CuTe runtime libraries supplied by the installed CuTe
package. The producer also needs the kernel source and CuTe compiler. Use
compatible host and GPU architectures and runtime libraries: embedding native
object bytes restricts the export's portability. For the experimental AOT
executable format, use matching JAX/jaxlib versions and a compatible device
configuration in the producer and consumer.

## Usage

### Serialized kernels

`make_softmax(N, dtype)` compiles the forward and backward kernels and returns a
differentiable callable that owns their serialized objects. Create it once and
reuse it with `jax.jit`:

```python
import jax
import jax.numpy as jnp
import jax_tvm_ffi
from examples.cutedsl.jax_softmax_serialized import make_softmax

jax_tvm_ffi.register_builtin_targets(platform="gpu")
softmax = make_softmax(N=1024, dtype=jnp.bfloat16)
x = jax.random.normal(jax.random.key(0), (32, 1024), dtype=jnp.bfloat16)
y = jax.jit(softmax)(x)
```

Use `make_softmax(..., allow_cuda_graph=True)` to allow XLA to capture the
forward and backward calls in CUDA graphs. The command-line example accepts
`--allow-cuda-graph` too. Capture remains subject to XLA's command-buffer settings.

### Registered kernels

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

## Tests

With CUDA-enabled JAX installed, run from the repository root:

```bash
pip install -e ".[cutedsl,orcjit]" pytest
JAX_PLATFORMS=cuda,cpu python -m pytest -q tests/test_cutlass.py
```

These tests compile and execute real CuTe kernels and exercise JAX export and
AOT executable round trips. They skip when a GPU or required optional
dependencies are unavailable. Install `flatbuffers` to include the JAX export
test; compile-and-run and serialized-executable tests do not require it.
Each flow is tested with CUDA graph capture enabled and disabled.

## Files

| File | Description |
|------|-------------|
| `softmax.py` | CuTeDSL kernel implementations (forward + backward) |
| `jax_softmax.py` | JAX integration of the softmax kernel |
| `jax_softmax_serialized.py` | JAX integration with object bytes embedded in StableHLO |
| `export_softmax.py` | Producer: write a JAX export and an AOT executable containing CuTe object bytes |
| `run_exported_softmax.py` | Consumer: restore either artifact without importing the kernel source |
