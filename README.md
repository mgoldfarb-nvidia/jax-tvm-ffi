# JAX TVM FFI

JAX TVM FFI is a library that enables seamless integration between JAX and TVM FFI,
allowing you to expose any function that is compatible with TVM FFI ABI to JAX.

## Installation

```bash
pip install .
```

## Quick Start

```bash
python run_example.py
```

## Testing

```bash
python -m pytest -vvs tests
```

## Usage

The API allows users to take a `tvm_ffi.Function` and connect it as a JAX FFI function.
The example below shows how to do this:

```python
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import tvm_ffi.cpp

# Create an inline C++ module
mod = tvm_ffi.cpp.load_inline(
    name="example",
    cpp_sources="""
        void add_one_cpu(tvm::ffi::TensorView x, tvm::ffi::TensorView y) {
            // implementation of a library function
            TVM_FFI_ICHECK(x.ndim() == 1) << "x must be a 1D tensor";
            DLDataType f32_dtype{kDLFloat, 32, 1};
            TVM_FFI_ICHECK(x.dtype() == f32_dtype) << "x must be a float tensor";
            TVM_FFI_ICHECK(y.ndim() == 1) << "y must be a 1D tensor";
            TVM_FFI_ICHECK(y.dtype() == f32_dtype) << "y must be a float tensor";
            TVM_FFI_ICHECK(x.size(0) == y.size(0)) << "x and y must have the same shape";
            for (int i = 0; i < x.size(0); ++i) {
              static_cast<float*>(y.data_ptr())[i] = static_cast<float*>(x.data_ptr())[i] + 1;
            }
        }
    """,
    functions=["add_one_cpu"],
)

# Register the function with JAX
jax_tvm_ffi.register_ffi_target("example.add_one_cpu", mod.add_one_cpu, platform="cpu")

# Use in JAX with JIT compilation
@jax.jit
def add_one_jax(x):
    return jax.ffi.ffi_call(
        "example.add_one_cpu",
        jax.ShapeDtypeStruct(x.shape, x.dtype),
        vmap_method="broadcast_all",
    )(x)

# Run the function
x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
result = add_one_jax(x)
print(f"Result: {result}")  # [2. 3. 4.]
```

### Custom Argument Specifications

You can customize how arguments are passed to your C++ functions:

```python
# Pass attributes as arguments
def my_function(eps, ret, input):
    # eps is passed as an attribute, x and y as tensors
    pass

jax_tvm_ffi.register_ffi_target(
    "my.function",
    my_function,
    arg_spec=["attrs.eps", "ret", "args"],  # eps from attrs, then rets, then args
    platform="cpu"
)

# Call with attributes
result = jax.ffi.ffi_call("my.function", output_shape)(x, y, eps=1e-5)
```

### Object-backed calls

Install the optional ORCJIT 0.1.1+ loader with `pip install jax-tvm-ffi[orcjit]`
once that wheel is published. Until then, use `uv sync --extra orcjit`; this
repository pins the coordinated
[upstream ORCJIT revision](https://github.com/apache/tvm-ffi/commit/c4a89570a9b361301eded1aff7933f5a7ff2f9f3).
`ffi_call_from_object` embeds a native relocatable object and entry-point name
in the StableHLO custom call. At executable instantiation, a shared TVM-FFI
ORCJIT session loads the object directly from the serialized bytes. JAX TVM FFI
weakly interns the resulting module by loader and payload SHA-256, then resolves
the exported host launcher from that module. Each executable strongly owns the
shared module, resolved function, and decoded argument mapping for every launch.
The cache does not extend module lifetime, so a module can unload after its last
executable and in-flight instantiation owner is destroyed.

CUTLASS DSL users can obtain the object bytes, exported name, and SHA-256
digest with `jax_tvm_ffi.cutlass.compile_to_object`. Object compilation uses a
bounded in-process cache keyed by precompiled artifact, target, and lowering
options; pass `no_cache=True` to force recompilation.

```python
from jax_tvm_ffi.cutlass import compile_to_object

serialized_function = compile_to_object(
    kernel,
    *compile_args,
    compile_options={"preserve-line-info": "true"},
)
call = jax_tvm_ffi.ffi_call_from_object(
    serialized_function.object_bytes,
    serialized_function.function_name,
    jax.ShapeDtypeStruct(output_shape, jnp.float32),
    platform="gpu",
    arg_spec=("args", "rets", "attrs.scale"),
    workspaces=(jax_tvm_ffi.Workspace(workspace_size),),
)
result = jax.jit(lambda x: call(x, scale=scale))(input)
```

The object should contain the TVM-FFI host export and any embedded device code,
such as a CUBIN. On Linux this is an ELF relocatable object; ORCJIT also accepts
the native object format on macOS and Windows. `ffi_call_from_payload` remains
available for other serialized formats through a registered loader with
signature `(Bytes) -> Module`. Call `clear_payload_module_cache()` to invalidate
weak lookup entries; live executables keep their modules valid.

Workspaces are trailing `uint8` outputs owned by XLA. They are hidden from the
Python result and passed to the loaded function as opaque pointers after its
ordinary input and output tensors. This is separate from
`use_last_output_for_alloc_workspace`, which provides TVM FFI's allocation
arena and does not pass the arena itself to the function.

### Python Callback

Because `tvm_ffi` supports Python functions out of the box, you can use the same
mechanism to register a Python function into the JAX system.
This feature is helpful for creating test cases and debugging.

```python
import numpy as np

def process_tensor(x, y):
    # Convert to NumPy arrays for processing
    x_np = np.from_dlpack(x)
    y_np = np.from_dlpack(y)
    y_np[:] = x_np + 1

jax_tvm_ffi.register_ffi_target(
    "process.tensor",
    process_tensor,
    platform="cpu",
    # Enable owned tensor access so from_dlpack can be called
    pass_owned_tensor=True
)
```

## CuTeDSL Integration

For an example of integrating high-performance [CuTeDSL](https://github.com/NVIDIA/cutlass) kernels
with JAX (including JIT, autodiff, and multi-GPU support), see [examples/cutedsl/](examples/cutedsl/).

```bash
pip install jax-tvm-ffi[cutedsl]
python -m examples.cutedsl.jax_softmax
```
