# pytorches-xpu

The Intel GPU backend for [PyTorches](https://github.com/tonyt123456/pytorches), shipped as its own
package so the main project does not have to bundle every vendor's plugin.

```
pip install pytorches-universe pytorches-xpu
```

There is nothing to import. PyTorches finds the plugin through the `pytorches.plugins` entry point,
checks its manifest, probes it in a child process (a broken driver cannot crash your program), and
registers `xpu:N` devices. `pytorches.doctor()` shows what was found.

It talks to the GPU through the OpenCL runtime that ships with Intel's graphics driver, loaded at
runtime, so no oneAPI or SDK install is needed. Devices that share system RAM (integrated Arc) are
reported as such so the placement planner can account for it.

## Layout

| Path | What |
|---|---|
| `src/lib.rs` | The plugin: implements the `pytorches-plugin-abi` vtable |
| `kernels/kernels.cl` | OpenCL C kernels, compiled by the driver at first use |
| `python/pytorches_xpu/` | The Python package that carries the library and `pytorches_plugin_xpu.json` (manifest) |
| `setup.py` | Runs `cargo build --release` and packages the library as a `py3-none-<platform>` wheel |
| `tests/conformance.rs` | Compares every op against the CPU plugin through the core; skips without an Intel GPU |
| `tests/limits.rs` | Allocation limits and allocator behavior (`--ignored` for the multi-GiB test) |

## Develop

```
cp .cargo/config.toml.example .cargo/config.toml   # build against a local ../pytorches checkout
cargo test                                         # builds the cdylib, runs the conformance suite
python -m build --wheel                            # dist/pytorches_xpu-*.whl
```

Debug knobs: `PYTORCHES_XPU_VERBOSE=1`, `PYTORCHES_XPU_ALLOC=host`, `PYTORCHES_XPU_BUILD_OPTS`.

The plugin depends on `pytorches-plugin-abi` (and, for tests, `pytorches-core` and
`pytorches-plugin-cpu`) from the main repository's `main` branch. ABI version 4 is required.

## Releasing

Tag `vX.Y.Z`; the workflow builds Windows and manylinux wheels and publishes to PyPI by trusted
publishing. Bump the version in `pyproject.toml` and `Cargo.toml` together.
