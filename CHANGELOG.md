# Changelog

All notable changes to TinyGuard-AI are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — 2026-11-21

Initial public research software release.

### Added

**Machine learning pipeline**
- Seeded synthetic vibration dataset generator (`ml/generate_dataset.py`) —
  3000 windows of 16 samples, balanced across one healthy and three fault
  scenarios, regenerating byte-for-byte.
- Integer feature extraction (`mean_absolute`, `rms`, `peak`) using truncating
  division and an integer square root, matching the firmware exactly.
- FP32 MLP training (3 → 8 → 1, ReLU, `lbfgs`, `random_state=42`).
- Symmetric per-tensor INT8 post-training quantization with INT32 biases and
  accumulators, calibrated on the training split only.
- Deterministic C header export for both the FP32 and INT8 parameter sets.
- `ml/header_params.py` — parses the committed C headers so verification runs
  against the parameters the firmware actually compiles.
- `ml/run_evaluation.py` — writes machine-readable metrics to `results/`:
  dataset summary, model metrics with baselines, quantization fidelity and
  footprint, and provenance with environment and artifact hashes. Every figure
  is computed at run time; none is hardcoded.

**Firmware**
- Bare-metal ARM Cortex-M4 firmware: vector table, `.data`/`.bss` init,
  custom linker script (256 K flash / 64 K RAM), semihosting output.
- FP32 and INT8 inference engines that run side by side and compare verdicts.
- Three synthetic demo windows: healthy, typical fault, and a deliberately
  severe out-of-distribution fault that demonstrates input quantizer
  saturation.
- Makefile targets for the cross build, QEMU execution, size and disassembly,
  plus the four host verification binaries and a `verify` target that runs the
  Python-versus-C cross-checks.

**Dashboard**
- Six-page Streamlit dashboard (`dashboard/`) covering overview, dataset,
  model, quantization, firmware and provenance. Live inference runs from the
  committed C headers rather than the pickle, so displayed figures are the
  values the firmware computes.

**Testing**
- 91 pytest tests requiring no C toolchain: dataset integrity, Python/C feature
  parity, quantization equivalence, byte-for-byte export determinism,
  firmware/Python constant synchronisation, metrics schema, documentation
  guards and dashboard behaviour.

**Project files**
- `README.md` with setup, reproduction steps, results and stated limitations.
- `LICENSE` (MIT), `requirements.txt` pinned to the versions that produced the
  committed artifacts, `.gitignore`, `.gitattributes`, `pytest.ini`.
- `CITATION.cff` for GitHub and Zenodo citation metadata.

### Verified in this release

- FP32 → INT8 prediction agreement: **100 %**, 0 mismatches across all 3000
  windows.
- Dataset and both generated C headers regenerate **byte-for-byte** from the
  committed sources.
- Evaluation metrics reproduce bit-for-bit across Windows and Linux and across
  three differing library stacks.
- Firmware builds warning-free under `-Wall -Wextra` and runs under QEMU
  (`mps2-an386`): 4,964 B flash, 0 B static RAM, ≈240 B peak stack.

### Known limitations

These bound every figure in this release and are stated throughout the
repository:

- The vibration dataset is **synthetic**. No physical accelerometer, motor or
  machine was measured.
- ARM execution is **QEMU emulation**. No physical microcontroller was
  programmed, run or timed.
- Reported classification accuracy reflects the **separability of the synthetic
  generator**, not demonstrated model capability — logistic regression and a
  depth-4 decision tree reach the same score, and a leakage-free grouped split
  does too.
- **No inference latency or energy figure is reported**, because neither can be
  obtained from emulation without a calibrated cycle model.
- The INT8 path is not floating-point-free: normalization and rescaling use
  float32, executed by soft-float routines on this FPU-less build. Only the
  multiply-accumulate and the class decision are integer.
- `INPUT_SCALE` is calibrated on training data, so inputs beyond
  `|z| ≈ 3.955` saturate and their reported logit is compressed. The severe
  demo window exercises this deliberately.

[1.0.0]: https://github.com/Quddos/tinyguard-ai/releases/tag/v1.0.0
