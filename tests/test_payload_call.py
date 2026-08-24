# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import jax
import jax_tvm_ffi
import numpy
import pytest
import tvm_ffi
import tvm_ffi.cpp
from jax import numpy as jnp

pytest.importorskip("tvm_ffi_orcjit")


@pytest.fixture(scope="module")
def object_bytes(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("payload_call")
    object_path = tvm_ffi.cpp.build_inline(
        name="payload_call_test",
        cpp_sources=r"""
            void add(tvm::ffi::Any increment, tvm::ffi::TensorView y,
                     tvm::ffi::TensorView x, void* workspace) {
              TVM_FFI_ICHECK(workspace != nullptr);
              TVM_FFI_ICHECK_EQ(x.ndim(), 1);
              TVM_FFI_ICHECK_EQ(x.shape()[0], y.shape()[0]);
              for (int64_t i = 0; i < x.shape()[0]; ++i) {
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] + increment.cast<int64_t>();
              }
            }
        """,
        functions=["add"],
        build_directory=str(tmp_path / "build"),
        output="payload_call_test.o",
    )
    return Path(object_path).read_bytes()


def test_object_call_embeds_object_and_passes_workspace(object_bytes, request):
    orcjit_load_object_function = tvm_ffi.get_global_func("tvm_ffi_orcjit.LoadObjectFunction")
    load_count = 0

    @tvm_ffi.register_global_func("tvm_ffi_orcjit.LoadObjectFunction", override=True)
    def load_object_function(payload, function_name):
        nonlocal load_count
        load_count += 1
        assert bytes(payload) == object_bytes
        return orcjit_load_object_function(payload, function_name)

    def restore_loader():
        tvm_ffi.register_global_func(
            "tvm_ffi_orcjit.LoadObjectFunction",
            orcjit_load_object_function,
            override=True,
        )

    request.addfinalizer(restore_loader)

    call = jax_tvm_ffi.ffi_call_from_object(
        object_bytes,
        "add",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        arg_spec=("attrs.increment", "rets", "args"),
        workspaces=(jax_tvm_ffi.Workspace(64),),
        vmap_method="broadcast_all",
    )
    add_two = jax.jit(lambda value: call(value, increment=2))
    add_three = jax.jit(lambda value: call(value, increment=3))
    x = jnp.arange(8, dtype=jnp.float32)

    stablehlo = str(add_two.lower(x).compiler_ir())
    assert "jax_tvm_ffi.payload_call" in stablehlo
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    numpy.testing.assert_equal(numpy.asarray(add_two(x)), numpy.asarray(x + 2))
    assert load_count == 1
    numpy.testing.assert_equal(numpy.asarray(add_three(x)), numpy.asarray(x + 3))
    assert load_count == 2


def test_object_call_rejects_missing_export(object_bytes):
    call = jax_tvm_ffi.ffi_call_from_object(
        object_bytes,
        "missing",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
    )

    with pytest.raises(Exception, match="does not export TVM FFI function 'missing'") as error:
        jax.jit(call)(jnp.arange(8, dtype=jnp.float32))
    assert "INVALID_ARGUMENT" in str(error.value)


def test_workspace_rejects_nonpositive_size():
    with numpy.testing.assert_raises_regex(ValueError, "must be positive"):
        jax_tvm_ffi.Workspace(0)
