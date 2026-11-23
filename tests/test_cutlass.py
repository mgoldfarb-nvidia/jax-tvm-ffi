# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the CuTe examples with real CUDA compilation and execution."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
from conftest import requires_gpu

pytestmark = requires_gpu


@pytest.fixture(scope="module", autouse=True)
def cutlass_dependencies():
    pytest.importorskip("cutlass")
    pytest.importorskip("tvm_ffi_orcjit")


def _run_example(module, *args):
    subprocess.run(
        [sys.executable, "-m", f"examples.cutedsl.{module}", *args],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            # Some CPU tests change JAX_PLATFORMS during collection.
            "JAX_PLATFORMS": "cuda",
            # The parent test process may already own a GPU memory pool.
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_FLAGS": "--xla_gpu_enable_command_buffer=CUSTOM_CALL "
            "--xla_gpu_graph_min_graph_size=1",
        },
        check=True,
        timeout=300,
    )


@pytest.mark.parametrize("allow_cuda_graph", [False, True])
def test_cutlass_compile_and_run(allow_cuda_graph):
    options = ("--allow-cuda-graph",) if allow_cuda_graph else ()
    _run_example("jax_softmax_serialized", *options)


@pytest.mark.parametrize(
    ("artifact_format", "artifact_name"),
    [("export", "softmax.jax"), ("executable", "softmax.pkl")],
)
@pytest.mark.parametrize("allow_cuda_graph", [False, True])
def test_cutlass_export_round_trip(tmp_path, artifact_format, artifact_name, allow_cuda_graph):
    if artifact_format == "export":
        pytest.importorskip("flatbuffers")
    options = ("--allow-cuda-graph",) if allow_cuda_graph else ()
    _run_example("export_softmax", tmp_path, "--format", artifact_format, *options)
    _run_example(
        "run_exported_softmax",
        tmp_path / artifact_name,
        "--format",
        artifact_format,
    )
