
#ifndef SENSOR_SIM_H
#define SENSOR_SIM_H

#include <stdint.h>

#define SENSOR_WINDOW_SIZE 16

typedef enum {
    SENSOR_NORMAL = 0,
    SENSOR_ABNORMAL = 1
} SensorScenario;

/*
 * Obtain a window of simulated vibration readings.
 * Values are acceleration deviations in mg.
 */
void sensor_sim_read(
    SensorScenario scenario,
    int32_t samples[SENSOR_WINDOW_SIZE]
);

#endif

