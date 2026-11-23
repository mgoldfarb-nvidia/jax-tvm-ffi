# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export a CuTe softmax as a JAX program and an AOT-compiled executable."""

import argparse
import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import jax_tvm_ffi
from jax import export
from jax.experimental import serialize_executable

from .jax_softmax_serialized import make_softmax


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_directory", type=Path)
    parser.add_argument("--format", choices=("export", "executable", "both"), default="both")
    args = parser.parse_args()

    target = jax_tvm_ffi.register_builtin_targets(platform="gpu")
    gpu = jax.devices("gpu")[0]
    args.output_directory.mkdir(parents=True, exist_ok=True)
    with jax.default_device(gpu):
        softmax = make_softmax(N=1024, dtype=jnp.float32)
        input_spec = jax.ShapeDtypeStruct((32, 1024), jnp.float32)
        softmax_jit = jax.jit(softmax)
        if args.format in ("export", "both"):
            exported = export.export(
                softmax_jit,
                platforms=("cuda",),
                disabled_checks=(export.DisabledSafetyCheck.custom_call(target),),
            )(input_spec)
            export_path = args.output_directory / "softmax.jax"
            export_path.write_bytes(exported.serialize())
            print(f"Exported JAX program: {export_path}")
        if args.format in ("executable", "both"):
            compiled = softmax_jit.lower(input_spec).compile()
            executable_path = args.output_directory / "softmax.pkl"
            executable_path.write_bytes(pickle.dumps(serialize_executable.serialize(compiled)))
            print(f"Compiled executable: {executable_path}")


if __name__ == "__main__":
    main()
