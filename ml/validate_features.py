
import math
import numpy as np

normal = [
    12, -10, 15, -13,
    8, -14, 11, -9,
    16, -12, 10, -11,
    14, -8, 9, -13
]

abnormal = [
    120, -110, 145, -130,
    180, -160, 135, -190,
    170, -140, 155, -175,
    190, -120, 165, -185
]

def calculate_features(samples):
    samples = np.array(samples, dtype=np.float64)

    mean_absolute = np.mean(np.abs(samples))
    rms = np.sqrt(np.mean(samples ** 2))
    peak = np.max(np.abs(samples))

    return mean_absolute, rms, peak

for name, samples in [
    ("Healthy", normal),
    ("Abnormal", abnormal)
]:
    mean_abs, rms, peak = calculate_features(samples)

    print(f"\n{name} motor")
    print(f"Mean absolute: {mean_abs:.3f} mg")
    print(f"RMS:           {rms:.3f} mg")
    print(f"Peak:          {peak:.3f} mg")
