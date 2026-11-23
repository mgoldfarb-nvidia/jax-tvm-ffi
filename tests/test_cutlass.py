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
    pytest.importorskip("flatbuffers")


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
        },
        check=True,
        timeout=300,
    )


def test_cutlass_compile_and_run():
    _run_example("jax_softmax_serialized")


@pytest.fixture(scope="module")
def exported_softmax(tmp_path_factory):
    artifact_directory = tmp_path_factory.mktemp("cutlass_export")
    _run_example("export_softmax", artifact_directory)
    return artifact_directory


@pytest.mark.parametrize(
    ("artifact_format", "artifact_name"),
    [("export", "softmax.jax"), ("executable", "softmax.pkl")],
)
def test_cutlass_export_round_trip(exported_softmax, artifact_format, artifact_name):
    _run_example(
        "run_exported_softmax",
        exported_softmax / artifact_name,
        "--format",
        artifact_format,
    )
