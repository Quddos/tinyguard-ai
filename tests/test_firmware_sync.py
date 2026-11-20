"""
Firmware / Python synchronisation tests.

No C toolchain is required: these tests parse the C sources as text and check
structural agreement with the Python side. They catch the class of bug where
someone edits one side of a duplicated constant and not the other.
"""

from __future__ import annotations

import re

import pytest

SENSOR_WINDOWS = {
    "normal_samples": [12, -10, 15, -13, 8, -14, 11, -9,
                       16, -12, 10, -11, 14, -8, 9, -13],
    "abnormal_typical_samples": [-12, 39, 39, -11, -57, -14, 29, 46,
                                 -11, -47, -39, 13, 74, 10, -25, -28],
    "abnormal_samples": [120, -110, 145, -130, 180, -160, 135, -190,
                         170, -140, 155, -175, 190, -120, 165, -185],
}


def c_array_values(source: str, name: str) -> list[int]:
    """Extract the integer initialiser of a named C array."""
    match = re.search(
        rf"{re.escape(name)}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\}}\s*;",
        source,
        flags=re.S,
    )
    assert match, f"could not find array {name!r} in the C source"
    return [int(tok) for tok in re.findall(r"-?\d+", match.group(1))]


@pytest.fixture(scope="module")
def sensor_sim_c(firmware_dir):
    return (firmware_dir / "sensor_sim.c").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def sensor_sim_h(firmware_dir):
    return (firmware_dir / "sensor_sim.h").read_text(encoding="utf-8")


def test_sensor_windows_match_python_mirror(sensor_sim_c):
    """
    ml/run_evaluation.py mirrors the firmware demo windows so the dashboard can
    display them. Those copies must stay identical to the C source.
    """
    from run_evaluation import FIRMWARE_WINDOWS

    mapping = {
        "SENSOR_NORMAL": "normal_samples",
        "SENSOR_ABNORMAL_TYPICAL": "abnormal_typical_samples",
        "SENSOR_ABNORMAL": "abnormal_samples",
    }

    for enum_name, array_name in mapping.items():
        c_values = c_array_values(sensor_sim_c, array_name)
        py_values = list(FIRMWARE_WINDOWS[enum_name])

        assert c_values == py_values, (
            f"{array_name} in sensor_sim.c differs from FIRMWARE_WINDOWS"
            f"[{enum_name!r}] in run_evaluation.py"
        )


def test_sensor_windows_have_expected_length(sensor_sim_c):
    for array_name in SENSOR_WINDOWS:
        assert len(c_array_values(sensor_sim_c, array_name)) == 16


def test_every_scenario_enum_is_handled(sensor_sim_c, sensor_sim_h):
    """Each enumerator must be selectable in sensor_sim_read()."""
    enumerators = re.findall(r"(SENSOR_[A-Z_]+)\s*=\s*\d+", sensor_sim_h)
    assert set(enumerators) == {
        "SENSOR_NORMAL", "SENSOR_ABNORMAL", "SENSOR_ABNORMAL_TYPICAL"
    }

    for name in enumerators:
        assert name in sensor_sim_c, f"{name} is never handled in sensor_sim.c"


def test_main_exercises_all_scenarios(firmware_dir):
    main_c = (firmware_dir / "main.c").read_text(encoding="utf-8")
    for name in ("SENSOR_NORMAL", "SENSOR_ABNORMAL_TYPICAL", "SENSOR_ABNORMAL"):
        assert f"monitor_motor({name})" in main_c, (
            f"main() does not run {name}"
        )


def test_model_dimensions_agree_between_headers(firmware_dir):
    """Both C headers declare MODEL_INPUTS/MODEL_HIDDEN; they must agree."""
    def defines(name):
        text = (firmware_dir / name).read_text(encoding="utf-8")
        return {
            k: int(v)
            for k, v in re.findall(r"#define\s+(MODEL_\w+)\s+(\d+)", text)
        }

    int8 = defines("model_int8_params.h")
    fp32 = defines("model_params.h")

    assert int8["MODEL_INPUTS"] == fp32["MODEL_INPUTS"] == 3
    assert int8["MODEL_HIDDEN"] == fp32["MODEL_HIDDEN"] == 8


def test_feature_accumulators_are_64_bit(firmware_dir):
    """
    Regression guard for the accumulator-width fix. A uint32 sum of squares
    overflows for large samples.
    """
    source = (firmware_dir / "signal_features.c").read_text(encoding="utf-8")

    assert "uint64_t sum_squares" in source, (
        "sum_squares must be 64-bit to avoid overflow on large samples"
    )
    assert "uint64_t sum_absolute" in source, (
        "sum_absolute must be 64-bit"
    )


def test_makefile_declares_all_host_tools(firmware_dir):
    """
    ml/verify_int8_c.py and ml/verify_embedded_model.py execute these binaries,
    so the Makefile must know how to build each one.
    """
    makefile = (firmware_dir / "Makefile").read_text(encoding="utf-8")

    for target in (
        "verify_int8_host",
        "verify_model_host",
        "test_model_host",
        "debug_int8_host",
    ):
        assert re.search(rf"^{target}:", makefile, flags=re.M), (
            f"Makefile has no rule to build {target}"
        )
        assert (firmware_dir / f"{target}.c").exists()

    assert re.search(r"^hosttools:", makefile, flags=re.M)


def test_verify_scripts_point_at_make_target(ml_dir):
    """
    If the host binary is missing the scripts must tell the user how to build
    it, rather than failing with a bare FileNotFoundError.
    """
    for name in ("verify_int8_c.py", "verify_embedded_model.py"):
        source = (ml_dir / name).read_text(encoding="utf-8")
        assert "make -C firmware hosttools" in source, (
            f"{name} should name the make target when the binary is absent"
        )


def test_limitations_are_stated_in_firmware_sources(firmware_dir):
    """
    The synthetic-data and QEMU-emulation limitations must remain visible in
    the firmware itself, not only in the README.
    """
    sensor_h = (firmware_dir / "sensor_sim.h").read_text(encoding="utf-8")
    main_c = (firmware_dir / "main.c").read_text(encoding="utf-8")

    assert "SYNTHETIC" in sensor_h.upper()
    assert "SYNTHETIC DATA" in main_c.upper()
    assert "QEMU" in main_c.upper()
