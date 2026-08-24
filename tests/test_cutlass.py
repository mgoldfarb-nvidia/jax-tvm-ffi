# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import sys
import types
from collections import OrderedDict

import pytest
from jax_tvm_ffi import cutlass


@pytest.fixture
def fake_cutlass(monkeypatch):
    state = types.SimpleNamespace(
        calls=[],
        object_count=0,
        serialized_artifact=b"mlir-and-metadata-a",
    )

    class ObjectArtifact:
        def __init__(self, data):
            self._data = data
            self.metadata = [types.SimpleNamespace(symbol_name="softmax")]

        def get_data(self):
            state.calls.append(("get_data",))
            return self._data

    class CompileCallable:
        def __getitem__(self, option):
            state.calls.append(("compile_option", option))
            return self

        def compile_to(self, target, function, *args):
            state.calls.append(("precompile", target, function, args))
            return types.SimpleNamespace(serialized=state.serialized_artifact)

    class CuteCompiler:
        def set_abi(self, abi):
            state.calls.append(("abi", abi))

        def set_device_target(self, arch):
            state.calls.append(("arch", arch))

        def add_compile_option(self, key, value):
            state.calls.append(("compiler_option", key, value))

        def compile_to(self, artifact, target):
            state.object_count += 1
            state.calls.append(("object", artifact.serialized, target))
            return ObjectArtifact(f"object-{state.object_count}".encode())

    def serialize_compilation_artifact(artifact):
        state.calls.append(("serialize", artifact.serialized))
        return artifact.serialized

    artifact_type = types.SimpleNamespace(PreCompiledMlir="mlir", Object="object")
    abi = types.SimpleNamespace(TvmFfi="tvm_ffi")
    compiler_module = types.ModuleType("cutlass.compiler")
    compiler_module.Abi = abi
    compiler_module.ArtifactType = artifact_type
    compiler_module.CuteCompiler = CuteCompiler
    compiler_module.serialize_compilation_artifact = serialize_compilation_artifact

    enable_tvm_ffi = object()
    cutlass_dsl_module = types.ModuleType("cutlass.cutlass_dsl")
    cutlass_dsl_module.EnableTVMFFI = enable_tvm_ffi

    cute_module = types.ModuleType("cutlass.cute")
    cute_module.compile = CompileCallable()

    cutlass_module = types.ModuleType("cutlass")
    cutlass_module.__path__ = []
    cutlass_module.compiler = compiler_module
    cutlass_module.cute = cute_module

    monkeypatch.setitem(sys.modules, "cutlass", cutlass_module)
    monkeypatch.setitem(sys.modules, "cutlass.compiler", compiler_module)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cute_module)
    monkeypatch.setitem(sys.modules, "cutlass.cutlass_dsl", cutlass_dsl_module)
    monkeypatch.setattr(cutlass, "_COMPILE_CACHE", OrderedDict())
    state.enable_tvm_ffi = enable_tvm_ffi
    return state


def test_compile_to_object_forwards_options_and_caches(fake_cutlass):
    function = lambda: None
    options = {
        "preserve-line-info": "true",
        "opt-level": "2",
    }

    result = cutlass.compile_to_object(
        function,
        "argument",
        gpu_arch="sm_90a",
        compile_options=options,
    )

    assert result == cutlass.SerializedFunction(object_bytes=b"object-1", function_name="softmax")
    assert result.sha256 == hashlib.sha256(b"object-1").hexdigest()
    assert fake_cutlass.calls == [
        ("compile_option", fake_cutlass.enable_tvm_ffi),
        ("precompile", "mlir", function, ("argument",)),
        ("serialize", b"mlir-and-metadata-a"),
        ("arch", "sm_90a"),
        ("compiler_option", "opt-level", "2"),
        ("compiler_option", "preserve-line-info", "true"),
        ("abi", "tvm_ffi"),
        ("object", b"mlir-and-metadata-a", "object"),
        ("get_data",),
    ]

    fake_cutlass.calls.clear()
    cached = cutlass.compile_to_object(
        function,
        "argument",
        gpu_arch="sm_90a",
        compile_options=dict(reversed(options.items())),
    )

    assert cached is result
    assert fake_cutlass.calls == [
        ("compile_option", fake_cutlass.enable_tvm_ffi),
        ("precompile", "mlir", function, ("argument",)),
        ("serialize", b"mlir-and-metadata-a"),
    ]


def test_compile_to_object_no_cache_bypasses_lookup_and_insertion(fake_cutlass):
    function = lambda: None
    cached = cutlass.compile_to_object(function, gpu_arch="sm_90a")

    fake_cutlass.calls.clear()
    uncached = cutlass.compile_to_object(function, gpu_arch="sm_90a", no_cache=True)

    assert uncached.object_bytes == b"object-2"
    assert all(call[0] != "serialize" for call in fake_cutlass.calls)

    fake_cutlass.calls.clear()
    assert cutlass.compile_to_object(function, gpu_arch="sm_90a") is cached
    assert [call[0] for call in fake_cutlass.calls] == [
        "compile_option",
        "precompile",
        "serialize",
    ]


def test_compile_to_object_cache_keys_artifact_arch_and_options(fake_cutlass):
    function = lambda: None
    first = cutlass.compile_to_object(function, gpu_arch="sm_90a")

    fake_cutlass.serialized_artifact = b"mlir-and-metadata-b"
    changed_artifact = cutlass.compile_to_object(function, gpu_arch="sm_90a")
    changed_options = cutlass.compile_to_object(
        function,
        gpu_arch="sm_90a",
        compile_options={"opt-level": "2"},
    )
    changed_arch = cutlass.compile_to_object(
        function,
        gpu_arch="sm_100a",
        compile_options={"opt-level": "2"},
    )

    assert [
        result.object_bytes for result in (first, changed_artifact, changed_options, changed_arch)
    ] == [b"object-1", b"object-2", b"object-3", b"object-4"]


@pytest.mark.parametrize(
    ("max_entries", "max_bytes"),
    [(1, 1024), (128, len(b"object-1"))],
)
def test_compile_to_object_cache_is_bounded(fake_cutlass, monkeypatch, max_entries, max_bytes):
    monkeypatch.setattr(cutlass, "_COMPILE_CACHE_MAX_ENTRIES", max_entries)
    monkeypatch.setattr(cutlass, "_COMPILE_CACHE_MAX_BYTES", max_bytes)
    function = lambda: None
    cutlass.compile_to_object(function, gpu_arch="sm_90a")

    fake_cutlass.serialized_artifact = b"mlir-and-metadata-b"
    cutlass.compile_to_object(function, gpu_arch="sm_90a")
    fake_cutlass.serialized_artifact = b"mlir-and-metadata-a"

    assert cutlass.compile_to_object(function, gpu_arch="sm_90a").object_bytes == b"object-3"
