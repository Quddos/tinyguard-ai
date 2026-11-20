# TinyGuard-AI

**An ARM Cortex-M4-Based TinyML Predictive Maintenance and Intelligent Sensor Monitoring System**

A complete, reproducible pipeline that trains a small neural network on vibration
features, quantizes it to INT8, exports it as C headers, and runs it as
bare-metal firmware on an ARM Cortex-M4 — with the integer inference engine
verified against the Python reference implementation.

---

## ⚠️ Scientific limitations — read first

These constraints bound every figure in this repository. They are not caveats
added at the end; they define what this project does and does not demonstrate.

| | Status |
|---|---|
| **Vibration data** | **Synthetic.** Generated analytically by [`ml/generate_dataset.py`](ml/generate_dataset.py). No accelerometer, motor or machine was ever measured. |
| **ARM execution** | **QEMU emulation** (`-M mps2-an386`). No physical microcontroller was programmed, run or timed. |
| **Accuracy figures** | Reflect the **separability of the synthetic generator**, not demonstrated model capability. See [Interpreting the accuracy figures](#interpreting-the-accuracy-figures). |
| **Latency / energy** | **Not measured.** Neither can be obtained from emulation without a calibrated cycle model, so no such numbers are reported. |

**This project does not claim physical sensor validation or on-hardware
testing.** Any downstream use — paper, report, presentation — must carry the
same two constraints.

---

## What this project actually demonstrates

The defensible contributions are **not** the classification score:

1. **A complete FP32 → INT8 quantization pipeline** for a 3→8→1 MLP: symmetric
   per-tensor weight quantization, INT32 biases, INT32 accumulators, and
   activation rescaling calibrated on the training split only.
2. **Bit-exact equivalence between the Python reference and the C engine.**
   `ml/verify_int8_c.py` reproduces C's `float32` arithmetic and
   round-away-from-zero semantics, then compares against the compiled binary
   sample by sample.
3. **Zero FP32 → INT8 prediction divergence** across all 3000 windows.
4. **A deterministic, auditable export path.** The dataset and both C headers
   regenerate byte-for-byte from the committed sources; the test suite enforces
   this.
5. **Integer-only feature extraction** in firmware that matches the Python
   feature extractor exactly, so the deployed model sees the distribution it
   was trained on.

---

## Results

All figures below are **computed** by [`ml/run_evaluation.py`](ml/run_evaluation.py)
and written to [`results/`](results/). Nothing is hardcoded. Regenerate with:

```powershell
python ml/run_evaluation.py
```

### Model vs baselines (held-out test split, n = 600)

| Model | Accuracy | F1 |
|---|---|---|
| Dummy (most frequent) | 0.500 | 0.000 |
| Hand threshold (`rms ≥ 50 or peak ≥ 100`) | 0.692 | 0.554 |
| Logistic regression | **1.000** | **1.000** |
| Decision tree (depth 4) | **1.000** | **1.000** |
| MLP 3→8→1 *(shipped)* | **1.000** | **1.000** |

### FP32 vs INT8 fidelity (all 3000 windows)

| Metric | Value |
|---|---|
| FP32 accuracy | 1.000 |
| INT8 accuracy | 1.000 |
| Prediction mismatches | **0** |
| Prediction agreement | **100.00 %** |
| Mean absolute logit error | 0.251 |
| 95th percentile logit error | 0.599 |
| Max absolute logit error | 1.263 |
| Windows saturating the input quantizer | 1 of 3000 (0.03 %) |

### Parameter footprint

| | FP32 | INT8 | Ratio |
|---|---|---|---|
| Weight tensors | 128 B | 32 B | **4.00×** |
| Biases | 36 B | 36 B (INT32) | 1.00× |
| StandardScaler | 24 B | 24 B (float32) | 1.00× |
| **Total** | **188 B** | **92 B** | **2.04×** |

> The often-quoted "4× smaller" applies only to the weight tensors. Biases stay
> INT32 and the scaler stays float32, so the honest whole-model figure is
> **2.04×**. Both are reported in `results/quantization.json`.

---

## Interpreting the accuracy figures

**The 100 % accuracy is a property of the dataset generator, not evidence that
the neural network is a good classifier.** Three independent checks establish
this:

1. **Simpler models tie it.** Logistic regression and a depth-4 decision tree
   also reach 1.000. An 8-unit MLP buys nothing measurable here.
2. **It is not train/test leakage.** The dataset holds only **1364 unique
   feature vectors across 3000 rows**, and **61.8 % of test rows share an exact
   feature vector with the training set** under the committed split. That is
   real leakage — but a leakage-free `GroupShuffleSplit` grouped by unique
   feature vector *still* yields 1.000 (n = 534, zero shared vectors). So
   leakage is not the cause.
3. **Generalisation across unseen fault types is near-perfect.** Leave-one-
   abnormal-scenario-out training scores 0.9988 on held-out `high_amplitude`
   (1 misclassified window out of 810) and 1.000 on held-out `impulsive` and
   `mixed_frequency`.

The actual cause: `generate_dataset.py` draws healthy and fault windows from
near-disjoint amplitude regimes. Individual features overlap slightly, but the
three jointly separate the classes almost perfectly.

**Consequence for any write-up:** report the quantization fidelity and the
C/Python equivalence as the result. Report accuracy only alongside the
baselines, and state that the task is separable by construction. A harder
dataset — overlapping regimes, or real measurements — is required before any
claim about model capability.

---

## Repository layout

```
ml/                              Python training, quantization and evaluation
  generate_dataset.py            Synthetic dataset generator (seeded, deterministic)
  train_model.py                 Trains the 3→8→1 MLP, saves .pkl artifacts
  export_model.py                FP32 weights  → firmware/model_params.h
  export_int8_model.py           INT8 weights  → firmware/model_int8_params.h
  header_params.py               Parses the committed C headers (drift guard)
  run_evaluation.py              Full evaluation → results/*.json
  quantization_test.py           Standalone INT8 vs FP32 comparison
  analyze_quantization.py        Per-tensor range and error analysis
  compare_models.py              Baseline comparison
  test_generalisation.py         Leave-one-scenario-out study
  verify_embedded_model.py       Python vs compiled C, FP32 path
  verify_int8_c.py               Python vs compiled C, INT8 path
  inspect_model.py               Prints architecture and a worked example
  inspect_features.py            Feature distribution summary
  validate_features.py           Feature extraction on the demo windows
  data/                          Committed synthetic dataset (3000 windows)
  models/                        Committed trained model and scaler

firmware/                        Bare-metal ARM Cortex-M4 firmware
  main.c                         Semihosting demo, runs FP32 and INT8 side by side
  startup.c  linker.ld           Vector table, .data/.bss init, 256K flash / 64K RAM
  sensor_sim.c/.h                Three synthetic vibration windows
  signal_features.c/.h           Integer mean-absolute / RMS / peak extraction
  tinyguard_model.c/.h           FP32 inference engine
  tinyguard_int8.c/.h            INT8 inference engine
  model_params.h                 Generated — do not edit by hand
  model_int8_params.h            Generated — do not edit by hand
  verify_int8_host.c             Host harness, INT8, stdin → stdout
  verify_model_host.c            Host harness, FP32, stdin → stdout
  test_model_host.c              Host smoke test
  debug_int8_host.c              Layer-by-layer INT8 dump
  Makefile                       Cross build, QEMU run, host tools, verify

dashboard/                       Streamlit visualization dashboard
  app.py                         Entry point and navigation
  lib/theme.py                   Validated chart palette, light and dark
  lib/data.py                    Cached loaders for dataset, metrics, C headers
  lib/inference.py               FP32 + INT8 inference from the committed headers
  lib/components.py              Stat tiles, meters, tables, chart+table pairing
  views/                         The six pages

tests/                           pytest suite (91 tests, no C toolchain needed)
results/                         Generated metrics artifacts (JSON)
```

---

## Requirements

**Python side** — Python 3.11+ and:

```powershell
python -m pip install -r requirements.txt
```

> **Reproducibility note:** the committed `.pkl` artifacts were serialised by
> **scikit-learn 1.9.1**. Loading them under another version emits
> `InconsistentVersionWarning`, and byte-exact header regeneration is only
> guaranteed on the pinned version. `results/provenance.json` records whether
> the running version matched.

**Firmware side** — only needed to build or emulate:

- `arm-none-eabi-gcc` (ARM GNU toolchain)
- `qemu-system-arm` (with `mps2-an386` machine support)
- `gcc` and `make` for the host verification binaries

---

## Reproducing everything

Each step is independent; the committed artifacts already reflect all of them.

### 1. Regenerate the dataset

```powershell
python ml/generate_dataset.py
```

### 2. Train the model

> **Warning:** this **overwrites** `ml/models/*.pkl`. Those committed artifacts
> were serialised by scikit-learn **1.9.1**; re-running under a different
> version rewrites them with that version's metadata, changing the SHA-256 in
> `results/provenance.json`. The learned weights are deterministic
> (`random_state=42`), so the exported headers are unaffected — but only run
> this if you intend to replace the artifacts.

```powershell
python ml/train_model.py
```

### 3. Export the C headers

```powershell
python ml/export_model.py
```

```powershell
python ml/export_int8_model.py
```

### 4. Run the full evaluation and write metrics

```powershell
python ml/run_evaluation.py
```

### 5. Run the test suite

```powershell
python -m pytest
```

### 6. Build and emulate the firmware

```powershell
make -C firmware
```

```powershell
make -C firmware run
```

```powershell
make -C firmware size
```

### 7. Launch the dashboard

```powershell
streamlit run dashboard/app.py
```

### 8. Verify the C engines against Python

Builds the four host binaries, then runs both cross-checks:

```powershell
make -C firmware verify
```

Or build the host tools alone:

```powershell
make -C firmware hosttools
```

**On Windows without GNU make/gcc**, the four host tools build with MSVC
directly. From a Developer PowerShell for VS:

```powershell
cd firmware; cl /nologo /std:c11 /O2 /D_CRT_SECURE_NO_WARNINGS /Fe:verify_int8_host.exe verify_int8_host.c tinyguard_int8.c
```

```powershell
cd firmware; cl /nologo /std:c11 /O2 /D_CRT_SECURE_NO_WARNINGS /Fe:verify_model_host.exe verify_model_host.c tinyguard_model.c
```

The verification scripts detect either the Unix-style or `.exe` binary name.
Only the ARM firmware itself requires `arm-none-eabi-gcc`.

---

## Dashboard

An interactive Streamlit dashboard over the real dataset and the committed
model outputs.

```powershell
streamlit run dashboard/app.py
```

It opens at `http://localhost:8501`.

| Page | Shows |
|---|---|
| **Overview** | Headline FP32/INT8 agreement, KPI row, baseline comparison, pipeline table |
| **Dataset** | Scenario composition, per-class feature distributions, the separability scatter, redundancy and leakage figures, raw 16-sample window inspector |
| **Model** | Baselines, both splits side by side with confusion matrices, per-scenario and leave-one-scenario-out meters |
| **Quantization** | FP32-vs-INT8 logit scatter, error distribution, input-headroom meter, scales, per-tensor error, footprint, header-consistency proof |
| **Firmware** | The three QEMU demo windows, the saturation comparison, and a live inference explorer with a layer-by-layer INT8 trace |
| **Provenance** | Generation record, environment, artifact SHA-256 hashes, reproduction commands, raw JSON |

**Design decisions worth knowing:**

- **It reads the C headers, not the `.pkl`.** All live inference in the
  dashboard runs on the constants in `firmware/model_params.h` and
  `firmware/model_int8_params.h`, reproducing C's `float32` arithmetic and
  round-away-from-zero cast. What the dashboard shows is what the firmware
  computes — it cannot display a different model, and it never triggers the
  scikit-learn version warning.
- **Nothing is hardcoded.** Every metric comes from `results/*.json` or from
  live inference. `tests/test_dashboard.py` enforces this.
- **The limitations banner is on every page**, and its text is read from
  `results/provenance.json` so it cannot drift from the artifacts.
- **Accuracy is never shown without baselines.** The Overview and Model pages
  put the dummy, threshold, logistic-regression and decision-tree scores
  directly beside the shipped model's.
- **Charts are accessibility-checked.** Three categorical colours, validated for
  colour-vision separation and contrast against the actual light (`#ffffff`)
  and dark (`#0e1117`) surfaces; separate dark-mode steps rather than an
  inverted palette; status colours reserved and always paired with an icon and
  a label; and every chart ships a table view so no value is reachable only by
  hovering.

> The dashboard needs `results/*.json`. If they are missing it says so and
> names the command to generate them rather than failing obscurely.

---

## Metrics artifacts

[`results/`](results/) is the machine-readable interface for the dashboard, the
results table above, and any figures.

| File | Contents |
|---|---|
| `index.json` | Headline figures, artifact hashes, limitations |
| `dataset_summary.json` | Class/scenario distributions, per-feature statistics, class overlap, redundancy, quantified split leakage |
| `model_metrics.json` | Shipped-model metrics, leakage-free grouped split, all baselines, per-scenario accuracy, leave-one-scenario-out |
| `quantization.json` | Scales, per-tensor error, FP32/INT8 fidelity, header-consistency proof, firmware demo window analysis, footprint |
| `provenance.json` | Timestamp, git commit, environment versions, SHA-256 of every artifact, limitations |

Every file is regenerated by `ml/run_evaluation.py`. The test suite asserts the
schema, internal consistency, and that the limitations and baselines are present.

---

## Model and quantization details

**Architecture** — 3 inputs → 8 hidden (ReLU) → 1 logit. Binary decision at
`logit ≥ 0`. Trained with `lbfgs`, `random_state=42`.

**Features** — `mean_absolute`, `rms`, `peak`, computed over a 16-sample
window using **truncating integer arithmetic and an integer square root**, so
[`ml/generate_dataset.py`](ml/generate_dataset.py) and
[`firmware/signal_features.c`](firmware/signal_features.c) agree exactly.

**Quantization** — symmetric per-tensor INT8 in `[-127, 127]`:

| Scale | Value |
|---|---|
| `INPUT_SCALE` | 0.0311435647309 |
| `W1_SCALE` | 0.0162149525632 |
| `ACC1_SCALE` | 0.000504991424759 |
| `HIDDEN_SCALE` | 0.177796744619 |
| `W2_SCALE` | 0.036833447484 |
| `ACC2_SCALE` | 0.00654886705577 |

Calibration uses the **training split only** — test data never informs a scale.

### Known limitation: input quantizer saturation

`INPUT_SCALE` is calibrated on training data, so the representable normalized
input range is `|z| ≤ 127 × INPUT_SCALE ≈ 3.955`. The dataset's maximum is
`|z| = 3.955`, so it just fits.

Inputs beyond that **clip at ±127**. The class decision survives, but the logit
is compressed. `firmware/sensor_sim.c` keeps a deliberately severe window to
demonstrate this:

| Window | Features | Saturated | FP32 logit | INT8 logit |
|---|---|---|---|---|
| `SENSOR_NORMAL` | (11, 11, 16) | no | −12.85 | −13.43 |
| `SENSOR_ABNORMAL_TYPICAL` | (30, 35, 74) | no | +43.58 | +43.24 |
| `SENSOR_ABNORMAL` | (154, 156, 190) | **yes** | +236.54 | **+128.33** |

The severe window's features exceed the dataset's abnormal maxima
(78, 87, 147), so two of three channels clip and the INT8 logit reads 46 % low.
**Use `SENSOR_ABNORMAL_TYPICAL` when quoting quantization fidelity.** This
behaviour is pinned by `tests/test_quantization.py`.

### Known limitation: 1-ULP scale representation gap

Each scale is computed in float64 and written as 12-significant-digit decimal
text. Parsing that text back as `float32` can land **1 ULP** away from casting
the float64 directly. This is a text-representation artifact, not exporter
drift, and it is provably inert: across all 3000 windows both scale sets give
**bit-identical quantized inputs, accumulators and predictions**, and the
smallest decision margin is **382 accumulator counts**. The integer tensors —
which alone determine the class — match bit-exactly. Verified in
`tests/test_quantization.py::test_scale_text_ulp_gap_is_inert`.

---

## Testing

91 tests, no C toolchain required (the C sources are checked structurally):

| File | Covers |
|---|---|
| `test_dataset.py` | Shape, balance, label consistency, redundancy, separability caveat |
| `test_features.py` | Python ↔ C integer feature parity, integer sqrt, accumulator width |
| `test_quantization.py` | Header integrity, FP32/INT8 equivalence, ULP-gap inertness, saturation |
| `test_determinism.py` | Dataset and both header exports regenerate byte-for-byte; no hardcoded metrics |
| `test_firmware_sync.py` | C/Python constant sync, Makefile targets, limitations present in firmware |
| `test_metrics_artifacts.py` | `results/*.json` schema, internal consistency, limitations and baselines retained |
| `test_documentation.py` | README states both limitations and makes no hardware/latency claim; LICENSE, pins and ignore rules intact |
| `test_dashboard.py` | Dashboard inference matches published metrics, reads headers not the pickle, no hardcoded figures, limitations on every page, palette validated |

```powershell
python -m pytest
```

---

## License

MIT — see [LICENSE](LICENSE).

**Recommendation:** MIT is the pragmatic default for research code meant to be
read, cited and reused, and imposes no conditions beyond attribution. Consider
switching if your situation differs:

- **Apache-2.0** — if you want an explicit patent grant (relevant if the
  quantization or firmware approach may be patentable, or if industry
  collaborators require it).
- **BSD-3-Clause** — if you want to forbid use of your name in endorsements.
- **CC-BY-4.0** *for the dataset specifically* — licenses are written for code;
  if the synthetic dataset is to be cited as a standalone research artifact, a
  data license alongside MIT is more conventional.

A public repository with no license defaults to *all rights reserved*, which
blocks exactly the academic reuse this project implies — so having one is the
important part.
