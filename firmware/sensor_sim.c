
#include "sensor_sim.h"

/*
 * Synthetic vibration windows. No physical sensor is involved.
 * See sensor_sim.h for the provenance and limitations of each window.
 */

/* Healthy: in-distribution. Features (11, 11, 16). */
static const int32_t normal_samples[SENSOR_WINDOW_SIZE] = {
    12, -10, 15, -13,
    8, -14, 11, -9,
    16, -12, 10, -11,
    14, -8, 9, -13
};

/*
 * Severe fault: OUT OF DISTRIBUTION. Features (154, 156, 190) exceed the
 * dataset's abnormal maxima (78, 87, 147) and saturate the INT8 input
 * quantizer. Retained to demonstrate saturation, not for accuracy claims.
 */
static const int32_t abnormal_samples[SENSOR_WINDOW_SIZE] = {
    120, -110, 145, -130,
    180, -160, 135, -190,
    170, -140, 155, -175,
    190, -120, 165, -185
};

/*
 * Typical fault: in-distribution, copied verbatim from dataset row 1387
 * (scenario high_amplitude). Features (30, 35, 74), no quantizer saturation.
 */
static const int32_t abnormal_typical_samples[SENSOR_WINDOW_SIZE] = {
    -12, 39, 39, -11,
    -57, -14, 29, 46,
    -11, -47, -39, 13,
    74, 10, -25, -28
};

void sensor_sim_read(
    SensorScenario scenario,
    int32_t samples[SENSOR_WINDOW_SIZE])
{
    const int32_t *source;

    switch (scenario) {
    case SENSOR_NORMAL:
        source = normal_samples;
        break;

    case SENSOR_ABNORMAL_TYPICAL:
        source = abnormal_typical_samples;
        break;

    case SENSOR_ABNORMAL:
    default:
        source = abnormal_samples;
        break;
    }

    for (uint32_t i = 0; i < SENSOR_WINDOW_SIZE; i++) {
        samples[i] = source[i];
    }
}
