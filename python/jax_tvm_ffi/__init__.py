# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""JAX TVM FFI Python package."""

import importlib
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import jax
import jax.ffi
import jax.numpy as jnp
import tvm_ffi


def _load_lib() -> tvm_ffi.Module:
    # first look at the directory of the current file
    file_dir = Path(__file__).resolve().parent

    if sys.platform.startswith("win32"):
        lib_dll_name = "jax_tvm_ffi.dll"
    elif sys.platform.startswith("darwin"):
        lib_dll_name = "jax_tvm_ffi.dylib"
    else:
        lib_dll_name = "jax_tvm_ffi.so"

    dirs = [file_dir, file_dir / ".." / ".." / "build"]
    for dir in dirs:
        if (dir / lib_dll_name).exists():
            lib_path = dir / lib_dll_name
            return tvm_ffi.load_module(str(lib_path))

    raise RuntimeError(f"Cannot find library: {lib_dll_name}")


_LIB = _load_lib()

_PAYLOAD_TARGET = "jax_tvm_ffi.payload_call"
_ORCJIT_PAYLOAD_LOADER = "tvm_ffi_orcjit"
_payload_registration_lock = threading.Lock()
_registered_payload_targets: set[str] = set()


@dataclass(frozen=True)
class Workspace:
    """A hidden device buffer passed to a payload function as an opaque pointer."""

    size_in_bytes: int

    def __post_init__(self) -> None:
        if self.size_in_bytes <= 0:
            raise ValueError("Workspace size_in_bytes must be positive")


def _get_dl_device_type(platform: str) -> int:
    """Get the dl device type from the platform."""
    if platform == "cpu":
        return tvm_ffi.DLDeviceType.kDLCPU
    elif platform == "gpu":
        return tvm_ffi.DLDeviceType.kDLCUDA
    else:
        raise ValueError(f"Unsupported platform: {platform}")


def _register_payload_target(platform: str) -> str:
    target = _PAYLOAD_TARGET
    with _payload_registration_lock:
        if platform in _registered_payload_targets:
            return target

        jax.ffi.register_ffi_target(
            target,
            {
                "prepare": jax.ffi.pycapsule(_LIB.payload_call_prepare_handler()),
                "execute": jax.ffi.pycapsule(_LIB.payload_call_execute_handler()),
            },
            platform=platform,
        )
        _registered_payload_targets.add(platform)
    return target


def ffi_call_from_payload(
    payload: bytes,
    payload_loader: str,
    function_name: str,
    result_shape_dtypes: Any,
    *,
    platform: str = "gpu",
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    """Build a JAX FFI call whose executable owns its compiled payload.

    ``payload_loader`` names a registered TVM global function with signature
    ``(Bytes, String) -> Function``. It is called during executable preparation
    when supported, or on the first execution otherwise. The returned function
    is cached by payload identity; the custom-call target itself is shared by
    every payload-backed call.

    Workspace buffers are appended to the custom call outputs, hidden from the
    returned JAX value, and passed to the loaded function as opaque pointers
    after its ordinary tensor arguments and results.
    """
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("payload must be nonempty bytes")
    if not payload_loader:
        raise ValueError("payload_loader must be nonempty")
    if not function_name:
        raise ValueError("function_name must be nonempty")
    if output_layouts is not None and len(output_layouts) != len(
        jax.tree.leaves(result_shape_dtypes)
    ):
        raise ValueError("output_layouts must describe only the visible results")

    target = _register_payload_target(platform)
    result_leaves, result_tree = jax.tree.flatten(result_shape_dtypes)
    workspace_results = tuple(
        jax.ShapeDtypeStruct((workspace.size_in_bytes,), jnp.uint8) for workspace in workspaces
    )
    all_results = (*result_leaves, *workspace_results)
    all_output_layouts = None
    if output_layouts is not None:
        all_output_layouts = (*output_layouts, *((0,) for _ in workspaces))

    call = jax.ffi.ffi_call(
        target,
        all_results,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=all_output_layouts,
        input_output_aliases=input_output_aliases,
    )

    def wrapped(*args: Any) -> Any:
        results = call(
            *args,
            payload=payload,
            payload_loader=payload_loader,
            function_name=function_name,
            device_type=int(_get_dl_device_type(platform)),
            num_workspace_outputs=len(workspaces),
        )
        visible_results = tuple(results[: len(result_leaves)])
        return jax.tree.unflatten(result_tree, visible_results)

    return wrapped


def ffi_call_from_object(
    object_bytes: bytes,
    function_name: str,
    result_shape_dtypes: Any,
    *,
    platform: str = "gpu",
    workspaces: Sequence[Workspace] = (),
    vmap_method: str | None = None,
    input_layouts: Sequence[Any] | None = None,
    output_layouts: Sequence[Any] | None = None,
    input_output_aliases: dict[int, int] | None = None,
) -> Callable[..., Any]:
    """Build a JAX FFI call backed by an in-memory native object file.

    The object bytes are serialized into the StableHLO custom call. At executable
    preparation, TVM-FFI ORCJIT loads the object directly from memory and resolves
    ``function_name`` as a TVM FFI export.
    """
    try:
        importlib.import_module("tvm_ffi_orcjit")
    except ImportError as error:
        raise ImportError(
            "ffi_call_from_object requires apache-tvm-ffi-orcjit; install jax-tvm-ffi[orcjit]"
        ) from error

    required_globals = (
        "tvm_ffi_orcjit.GlobalDefaultSession",
        "tvm_ffi_orcjit.SessionLoadModule",
    )
    missing_globals = [
        name
        for name in required_globals
        if tvm_ffi.get_global_func(name, allow_missing=True) is None
    ]
    if missing_globals:
        raise RuntimeError(
            "The installed apache-tvm-ffi-orcjit does not support in-memory module loading; "
            f"missing {', '.join(missing_globals)}"
        )

    return ffi_call_from_payload(
        object_bytes,
        _ORCJIT_PAYLOAD_LOADER,
        function_name,
        result_shape_dtypes,
        platform=platform,
        workspaces=workspaces,
        vmap_method=vmap_method,
        input_layouts=input_layouts,
        output_layouts=output_layouts,
        input_output_aliases=input_output_aliases,
    )


def register_ffi_target(
    name: str,
    function: tvm_ffi.Function,
    arg_spec: Optional[list[str]] = None,
    platform: str = "cpu",
    *,
    allow_cuda_graph: bool = False,
    pass_owned_tensor: bool = False,
    use_last_output_for_alloc_workspace: bool = False,
) -> Callable:
    """Function to register a ffi target for jax with tvm_ffi.Function

    Parameters
    ----------
    name: str
        The name of the ffi target

    function: tvm_ffi.Function
        The function to register

    arg_spec: Optional[list[str]]
        The arg spec of the function specifying how to map inputs to arguments to the function
        can be "args", "rets", or "attrs.<key>" for attributes

    platform: str
        The platform of the ffi target

    allow_cuda_graph: bool
        Whether the function can be used in cuda graph capture

    pass_owned_tensor: bool
        Whether the function can pass owned tensor to the function. This flag can be helpful
        in python callback when we want to further call from_dlpack on the tensor.
        However, the function should not retain the tensor after the function call.

    use_last_output_for_alloc_workspace: bool
        Whether to use the last output as allocation workspace for alloc_tensor() calls.
        If True, the user must provide workspace as the LAST element in result_shape_dtypes
        when calling the function. The workspace buffer should be jax.ShapeDtypeStruct((size,), jnp.uint8).

    Returns
    -------
    Callable
        The registered FFI capsule that can be used with jax.ffi.ffi_call

    Notes
    -----
    The arg_spec specifies how the inputs, outputs, and attributes are mapped to the
    underlying call to ffi function `fun`. Some examples:

    - `["args", "rets"]` maps to `fun(*args, *rets)`
    - `["attrs.key0", "args"]` maps to `fun(attrs["key0"], *args, *rets)`
    - `["attrs.key0", "args", "attrs.key1"]` maps to `fun(attrs["key0"], *args, attrs["key1"])`

    Workspace Allocation
    --------------------
    When use_last_output_for_alloc_workspace=True, the kernel can call alloc_tensor() for temporary memory.
    The user must explicitly provide workspace in the output tuple at call time:

    Example with workspace:
        >>> # 1. Register with workspace flag
        >>> my_kernel = register_ffi_target(
        ...     "my_kernel",
        ...     tvm_module.my_kernel,
        ...     use_last_output_for_alloc_workspace=True,
        ...     platform="gpu",
        ...     allow_cuda_graph=True
        ... )
        ...
        >>> # 2. Call with workspace in output tuple (LAST position)
        >>> workspace_size = 1024 * 4  # Compute based on your needs, each allocation is aligned to 128 bytes
        >>> result = jax.ffi.ffi_call(
        ...     "my_kernel",
        ...     (
        ...         jax.ShapeDtypeStruct(output_shape, jnp.float32),  # Actual output
        ...         jax.ShapeDtypeStruct((workspace_size,), jnp.uint8)  # Workspace (LAST!)
        ...     ),
        ...     input_array
        ... )
        >>> output = result[0]  # Strip workspace from results

    """
    # by default, we use the arg spec "args" and "rets"
    arg_spec = arg_spec if arg_spec is not None else ["args", "rets"]
    dl_device_type = _get_dl_device_type(platform)
    traits = 1 if allow_cuda_graph else 0
    fn = jax.ffi.pycapsule(
        _LIB.register_tvm_ffi_handler(
            function,
            arg_spec,
            dl_device_type,
            traits,
            pass_owned_tensor,
            use_last_output_for_alloc_workspace,
        )
    )
    jax.ffi.register_ffi_target(name, fn, platform=platform)
    return fn


def registered_count() -> int:
    """Get the number of registered functions"""
    return _LIB.registered_count()


def get_last_workspace_peak() -> int:
    """Get peak workspace usage from the last FFI call that used workspace.

    Returns
    -------
    int
        Peak workspace bytes used, or 0 if no workspace was used
    """
    return _LIB.get_last_workspace_peak()
