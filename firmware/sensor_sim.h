
#ifndef SENSOR_SIM_H
#define SENSOR_SIM_H

#include <stdint.h>

#define SENSOR_WINDOW_SIZE 16

/*
 * SCENARIOS
 *
 * SENSOR_NORMAL
 *     Healthy window. Features (11, 11, 16) lie inside the training
 *     distribution for the healthy class.
 *
 * SENSOR_ABNORMAL_TYPICAL
 *     Fault window taken verbatim from the synthetic dataset
 *     (ml/data/tinyguard_vibration_dataset.csv, row 1387, high_amplitude).
 *     Features (30, 35, 74) sit near the median of the abnormal class, so the
 *     INT8 input quantizer does NOT saturate and the INT8 logit tracks the
 *     FP32 logit closely. This is the scenario to quote when reporting
 *     quantization fidelity.
 *
 * SENSOR_ABNORMAL
 *     Severe fault window retained from the original implementation. Its
 *     features (154, 156, 190) are OUTSIDE the training distribution: the
 *     abnormal class maxima in the dataset are (78, 87, 147). All three
 *     normalized inputs exceed the calibrated INT8 input range, so two of the
 *     three channels clip at +127. The class decision is still correct, but
 *     the returned logit is heavily compressed (FP32 +236.5 vs INT8 +128.3).
 *     Kept deliberately as a demonstration of quantizer saturation on
 *     out-of-distribution input - do not use it to characterise accuracy.
 */
typedef enum {
    SENSOR_NORMAL = 0,
    SENSOR_ABNORMAL = 1,
    SENSOR_ABNORMAL_TYPICAL = 2
} SensorScenario;

/*
 * Obtain a window of simulated vibration readings.
 *
 * Values are acceleration deviations in mg. These are SYNTHETIC signals -
 * no physical accelerometer is read at any point.
 */
void sensor_sim_read(
    SensorScenario scenario,
    int32_t samples[SENSOR_WINDOW_SIZE]
);

#endif
