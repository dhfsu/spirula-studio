# spirula-sfm

Standalone Vulkan Structure-from-Motion reconstruction.

The program accepts images and writes COLMAP-compatible sparse models. The
pipeline includes feature extraction, matching, geometric verification,
incremental or bottom-up mapping, model merging, bundle adjustment, and resume
support. It has no training engine, GUI, web viewer, Python runtime, CUDA
backend, or learned segmentation models.

## Requirements

- C++17 compiler
- Vulkan loader and headers
- Ninja
- Slang compiler; CMake downloads the pinned version when `slangc` is absent

## Build

```bash
./build_develop.bash -DSS_DEFAULT_LANG=en
```

The executable is `build/spirula-sfm`. Use `--help` for commands and options.

To select a different build directory:

```bash
BUILD_DIR=build-release ./build_develop.bash
```

The generated SPIR-V is embedded in the executable. No Python interpreter is
needed to configure or build the project.

## Layout

- `src/sfm/`: SfM pipeline, Vulkan shaders, and native tests
- `src/core/`: small host utilities shared by SfM
- `src/data/`: YAML/JSON readers used by manifests
- `src/i18n/`: localized CLI and log messages
- `src/app/`: the standalone command entry point
- `cmake/`: Vulkan, Slang, and SfM build modules

See [src/sfm/README.md](src/sfm/README.md) for the pipeline and file formats.
