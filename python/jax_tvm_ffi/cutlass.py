# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CUTLASS DSL compilation helpers for JAX TVM FFI."""

import ctypes
import hashlib
import importlib
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax

from . import SerializedFunction


@dataclass(frozen=True)
class _CompileCacheKey:
    precompiled_sha256: bytes
    gpu_arch: str
    compile_options: tuple[tuple[str, str], ...]


_COMPILE_CACHE_MAX_ENTRIES = 128
_COMPILE_CACHE_MAX_BYTES = 256 * 1024**2
_COMPILE_CACHE: OrderedDict[_CompileCacheKey, SerializedFunction] = OrderedDict()
_COMPILE_LOCK = threading.Lock()
_RUNTIME_LOCK = threading.Lock()
_RUNTIME_LIBRARY_HANDLES: dict[str, ctypes.CDLL] = {}


def load_runtime() -> None:
    """Load CUTLASS DSL runtime libraries for ORCJIT symbol resolution.

    The handles remain alive for the process lifetime because serialized
    modules can run destructors after their last JAX executable is released.
    """
    with _RUNTIME_LOCK:
        if _RUNTIME_LIBRARY_HANDLES:
            return
        try:
            cutlass_runtime = importlib.import_module("cutlass.runtime")
        except ImportError as error:
            raise ImportError(
                "load_runtime requires CuTe DSL; install jax-tvm-ffi[cutedsl]"
            ) from error

        runtime_paths = cutlass_runtime.find_runtime_libraries(enable_tvm_ffi=False)
        if not runtime_paths:
            raise RuntimeError("CuTe DSL did not provide a runtime library")
        mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)
        handles: dict[str, ctypes.CDLL] = {}
        for path_like in runtime_paths:
            path = os.fspath(path_like)
            try:
                handles[path] = ctypes.CDLL(path, mode=mode)
            except OSError as error:
                raise RuntimeError(f"Failed to load CuTe DSL runtime library {path!r}") from error
        _RUNTIME_LIBRARY_HANDLES.update(handles)


def _target_arch(gpu_arch: str | None) -> str:
    if gpu_arch:
        return gpu_arch
    override = os.environ.get("CUTE_DSL_ARCH")
    if override:
        return override
    major, minor = map(int, jax.devices("gpu")[0].compute_capability.split("."))
    return f"sm_{major}{minor}{'a' if major >= 9 else ''}"


def _insert_compile_cache(key: _CompileCacheKey, value: SerializedFunction) -> None:
    if len(value.object_bytes) > _COMPILE_CACHE_MAX_BYTES:
        return
    _COMPILE_CACHE[key] = value
    cache_size = sum(len(entry.object_bytes) for entry in _COMPILE_CACHE.values())
    while len(_COMPILE_CACHE) > _COMPILE_CACHE_MAX_ENTRIES or cache_size > _COMPILE_CACHE_MAX_BYTES:
        _, evicted = _COMPILE_CACHE.popitem(last=False)
        cache_size -= len(evicted.object_bytes)


def compile_to_object(
    function: Callable[..., None],
    *compile_args: Any,
    gpu_arch: str | None = None,
    compile_options: Mapping[str, str] | None = None,
    no_cache: bool = False,
) -> SerializedFunction:
    """Compile a CuTe DSL function to a serialized TVM FFI object.

    Args:
        function: A ``@cute.jit`` function or callable object.
        *compile_args: Representative arguments describing its runtime signature.
        gpu_arch: Optional CuTe target such as ``sm_90a``. Defaults to
            ``CUTE_DSL_ARCH`` and then the first JAX GPU.
        compile_options: Lowering options passed to ``CuteCompiler.add_compile_option``.
            Entries override the compiler defaults. This helper always emits
            the TVM FFI ABI.
        no_cache: Bypass the process-wide compiled-object cache.

    Returns:
        The object bytes, exported TVM FFI function name, and SHA-256 digest.

    Notes:
        This helper uses CuTe DSL's experimental fine-grained compilation API.
        It requires the coordinated compiler change that makes CUDA TVM FFI
        objects own lazy initialization and teardown; public CuTe DSL 4.6 does
        not include that behavior. CuTe runtime libraries are loaded globally
        so ORCJIT can resolve the object's host-runtime symbols.
        Cache keys include the serialized PreCompiledMlir artifact, resolved
        GPU architecture, and lowering options. The cache evicts least-recently
        used objects when it reaches its entry or byte limit.
    """
    try:
        cutlass_compiler = importlib.import_module("cutlass.compiler")
        cute = importlib.import_module("cutlass.cute")
        cutlass_dsl = importlib.import_module("cutlass.cutlass_dsl")
    except ImportError as error:
        raise ImportError(
            "compile_to_object requires the coordinated CuTe DSL source build; "
            "install jax-tvm-ffi[cutedsl] and replace its public 4.6 compiler "
            "with that build"
        ) from error

    resolved_arch = _target_arch(gpu_arch)
    normalized_options = tuple(sorted((compile_options or {}).items()))

    with _COMPILE_LOCK:
        load_runtime()
        precompiled = cute.compile[cutlass_dsl.EnableTVMFFI].compile_to(
            cutlass_compiler.ArtifactType.PreCompiledMlir,
            function,
            *compile_args,
        )
        cache_key = None
        if not no_cache:
            # Function metadata controls wrapper emission, so hash the complete
            # artifact serialization rather than the MLIR payload alone.
            serialized_artifact = cutlass_compiler.serialize_compilation_artifact(precompiled)
            cache_key = _CompileCacheKey(
                precompiled_sha256=hashlib.sha256(serialized_artifact).digest(),
                gpu_arch=resolved_arch,
                compile_options=normalized_options,
            )
            cached = _COMPILE_CACHE.get(cache_key)
            if cached is not None:
                _COMPILE_CACHE.move_to_end(cache_key)
                return cached

        compiler = cutlass_compiler.CuteCompiler()
        compiler.set_device_target(resolved_arch)
        for option_name, option_value in normalized_options:
            compiler.add_compile_option(option_name, option_value)
        compiler.set_abi(cutlass_compiler.Abi.TvmFfi)
        artifact = compiler.compile_to(precompiled, cutlass_compiler.ArtifactType.Object)
        result = SerializedFunction(
            object_bytes=artifact.get_data(),
            function_name=artifact.metadata[0].symbol_name,
        )
        if cache_key is not None:
            _insert_compile_cache(cache_key, result)
        return result
