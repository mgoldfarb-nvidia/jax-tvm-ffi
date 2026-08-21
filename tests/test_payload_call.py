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

tvm_ffi_orcjit = pytest.importorskip("tvm_ffi_orcjit")


def test_object_call_embeds_object_and_passes_workspace(tmp_path):
    object_path = tvm_ffi.cpp.build_inline(
        name="payload_call_test",
        cpp_sources=r"""
            void add_one(tvm::ffi::TensorView x, tvm::ffi::TensorView y, void* workspace) {
              TVM_FFI_ICHECK(workspace != nullptr);
              TVM_FFI_ICHECK_EQ(x.ndim(), 1);
              TVM_FFI_ICHECK_EQ(x.shape()[0], y.shape()[0]);
              for (int64_t i = 0; i < x.shape()[0]; ++i) {
                static_cast<float*>(y.data_ptr())[i] =
                    static_cast<float*>(x.data_ptr())[i] + 1;
              }
            }
        """,
        functions=["add_one"],
        build_directory=str(tmp_path / "build"),
        output="payload_call_test.o",
    )
    object_bytes = Path(object_path).read_bytes()
    session = tvm_ffi_orcjit.ExecutionSession()
    load_count = 0

    @tvm_ffi.register_global_func("tvm_ffi_orcjit.GlobalDefaultSession", override=True)
    def get_default_session():
        return session

    @tvm_ffi.register_global_func("tvm_ffi_orcjit.SessionLoadModule", override=True)
    def load_module(_session, objects, _name):
        nonlocal load_count
        load_count += 1
        assert len(objects) == 1
        assert bytes(objects[0]) == object_bytes

        # apache-tvm-ffi-orcjit 0.1.0 only accepts a path. This adapter exercises
        # the same globals as its upstream in-memory SessionLoadModule API.
        compatibility_path = tmp_path / "payload_from_hlo.o"
        compatibility_path.write_bytes(bytes(objects[0]))
        module = session.create_library()
        module.add(compatibility_path)
        return module

    call = jax_tvm_ffi.ffi_call_from_object(
        object_bytes,
        "add_one",
        jax.ShapeDtypeStruct((8,), jnp.float32),
        platform="cpu",
        workspaces=(jax_tvm_ffi.Workspace(64),),
        vmap_method="broadcast_all",
    )
    compiled_call = jax.jit(call)
    x = jnp.arange(8, dtype=jnp.float32)

    stablehlo = str(compiled_call.lower(x).compiler_ir())
    assert "jax_tvm_ffi.payload_call" in stablehlo
    numpy.testing.assert_equal(numpy.asarray(compiled_call(x)), numpy.asarray(x + 1))
    numpy.testing.assert_equal(numpy.asarray(compiled_call(x)), numpy.asarray(x + 1))
    assert load_count == 1


def test_workspace_rejects_nonpositive_size():
    with numpy.testing.assert_raises_regex(ValueError, "must be positive"):
        jax_tvm_ffi.Workspace(0)
