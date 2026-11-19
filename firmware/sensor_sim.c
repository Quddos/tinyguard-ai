
#include "sensor_sim.h"

static const int32_t normal_samples[SENSOR_WINDOW_SIZE] = {
    12, -10, 15, -13,
    8, -14, 11, -9,
    16, -12, 10, -11,
    14, -8, 9, -13
};

static const int32_t abnormal_samples[SENSOR_WINDOW_SIZE] = {
    120, -110, 145, -130,
    180, -160, 135, -190,
    170, -140, 155, -175,
    190, -120, 165, -185
};

void sensor_sim_read(
    SensorScenario scenario,
    int32_t samples[SENSOR_WINDOW_SIZE])
{
    const int32_t *source;

    if (scenario == SENSOR_NORMAL) {
        source = normal_samples;
    } else {
        source = abnormal_samples;
    }

    for (uint32_t i = 0; i < SENSOR_WINDOW_SIZE; i++) {
        samples[i] = source[i];
    }
}
