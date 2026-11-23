# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JAX integration for serialized CuTe DSL softmax kernels.

``make_softmax`` compiles forward and backward kernels and returns a callable
that embeds their object bytes in StableHLO custom calls. Each custom-call
instance loads an ORCJIT module and retains it with its resolved TVM FFI
function for the executable's lifetime.

Usage:
    python -m examples.cutedsl.jax_softmax_serialized
"""

import argparse
from collections.abc import Callable

import cutlass
import jax
import jax.numpy as jnp
import jax_tvm_ffi
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax_tvm_ffi.cutlass import SerializedFunction, compile_to_object

from .jax_softmax import (
    DType,
    _get_cutlass_dtype,
    _make_example_tensor,
)
from .softmax import SoftmaxBackward, SoftmaxForward


def _compile_softmax_forward_object(
    dtype: type[cutlass.Numeric], N: int, M_example: int = 32
) -> SerializedFunction:
    example_x = _make_example_tensor(M_example, N, dtype)
    example_o = _make_example_tensor(M_example, N, dtype)
    return compile_to_object(SoftmaxForward(dtype, N), example_x, example_o)


def _compile_softmax_backward_object(
    dtype: type[cutlass.Numeric], N: int, M_example: int = 32
) -> SerializedFunction:
    example_dy = _make_example_tensor(M_example, N, dtype)
    example_y = _make_example_tensor(M_example, N, dtype)
    example_dx = _make_example_tensor(M_example, N, dtype)
    return compile_to_object(SoftmaxBackward(dtype, N), example_dy, example_y, example_dx)


def make_softmax(
    N: int, dtype: DType = jnp.bfloat16, *, allow_cuda_graph: bool = False
) -> Callable[[Array], Array]:
    """Compile a softmax callable for two-dimensional arrays of shape ``(M, N)``."""
    dtype = jnp.dtype(dtype)
    cutlass_dtype = _get_cutlass_dtype(dtype)
    forward = _compile_softmax_forward_object(cutlass_dtype, N)
    backward = _compile_softmax_backward_object(cutlass_dtype, N)

    @jax.custom_vjp
    def softmax_fn(x: Array) -> Array:
        if x.ndim != 2 or x.shape[1] != N:
            raise ValueError(f"Expected shape (M, {N}), got {x.shape}")
        if x.dtype != dtype:
            raise ValueError(f"Expected dtype {dtype}, got {x.dtype}")
        call = jax_tvm_ffi.ffi_call_from_serialized(
            forward,
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            platform="gpu",
            allow_cuda_graph=allow_cuda_graph,
            arg_spec=("args", "rets"),
            vmap_method="broadcast_all",
        )
        return call(x)

    def softmax_fwd(x: Array) -> tuple[Array, Array]:
        y = softmax_fn(x)
        return y, y

    def softmax_bwd(y: Array, g: Array) -> tuple[Array]:
        call = jax_tvm_ffi.ffi_call_from_serialized(
            backward,
            jax.ShapeDtypeStruct(g.shape, g.dtype),
            platform="gpu",
            allow_cuda_graph=allow_cuda_graph,
            arg_spec=("args", "rets"),
            vmap_method="broadcast_all",
        )
        return (call(g, y),)

    softmax_fn.defvjp(softmax_fwd, softmax_bwd)
    return softmax_fn


def main() -> None:
    """Demonstrate serialized CuTe DSL softmax integration with JAX."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cuda-graph", action="store_true")
    args = parser.parse_args()
    jax_tvm_ffi.register_builtin_targets(platform="gpu")
    print("=" * 60)
    print("Serialized CuTe DSL Softmax + JAX Integration Demo")
    print("=" * 60)

    N, dtype = 1024, jnp.bfloat16
    print(f"\nConfig: N={N}, dtype={dtype}")

    print("\n[1] Compiling and serializing CuTe DSL kernels...")
    softmax = make_softmax(N=N, dtype=dtype, allow_cuda_graph=args.allow_cuda_graph)
    print("    Done!")

    print("\n[2] Testing JIT compilation...")
    x = jax.random.normal(jax.random.key(42), (32, N), dtype=dtype)
    softmax_jit = jax.jit(softmax)
    y_ref = jax.nn.softmax(x, axis=-1)
    # XLA warms up once before capture; exercise capture and replay as well.
    for _ in range(3):
        y = softmax_jit(x)
        max_diff = float(jnp.abs(y - y_ref).max())
        assert max_diff < 1e-3, f"JIT failed: max_diff={max_diff}"
    print(f"    PASS (max_diff={max_diff:.2e})")

    print("\n[3] Testing autodiff (jax.grad)...")
    backward_jit = jax.jit(jax.grad(lambda value: softmax(value).sum()))
    for _ in range(3):
        dx = backward_jit(x).block_until_ready()
    print(f"    PASS (dx shape={dx.shape})")

    print("\n[4] Testing shard_map...")
    devices = jax.devices("gpu")
    if len(devices) >= 2:
        mesh = Mesh(np.array(devices[:2]), axis_names=("batch",))
        x_sharded = jax.device_put(x, NamedSharding(mesh, P("batch", None)))
        y_sharded = jax.shard_map(
            softmax,
            mesh=mesh,
            in_specs=(P("batch", None),),
            out_specs=P("batch", None),
        )(x_sharded)
        max_diff = float(jnp.abs(y_sharded - y_ref).max())
        assert max_diff < 1e-3, f"shard_map failed: max_diff={max_diff}"
        print(f"    PASS (2 GPUs, max_diff={max_diff:.2e})")
    else:
        print(f"    SKIP (requires 2+ GPUs, found {len(devices)})")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
