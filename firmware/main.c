
#include <stdint.h>
#include "model_int8_params.h"

#include "sensor_sim.h"
#include "signal_features.h"
#include "tinyguard_model.h"
#include "tinyguard_int8.h"

/* ARM semihosting interface */

static int semihost_call(
    uint32_t operation,
    const void *parameter)
{
    register uint32_t r0 __asm("r0") = operation;
    register const void *r1 __asm("r1") = parameter;

    __asm volatile(
        "bkpt 0xAB"
        : "+r"(r0)
        : "r"(r1)
        : "memory"
    );

    return (int)r0;
}

static void print(const char *message)
{
    semihost_call(0x04, message);
}

static void print_number(uint32_t value)
{
    char buffer[11];
    uint32_t position = 10;

    buffer[position] = '\0';

    do {
        buffer[--position] = '0' + (value % 10);
        value /= 10;
    } while (value > 0);

    print(&buffer[position]);
}

/* Compare the original FP32 and new INT8 models */

static void monitor_motor(SensorScenario scenario)
{
    int32_t samples[SENSOR_WINDOW_SIZE];
    SignalFeatures features;

    sensor_sim_read(scenario, samples);

    calculate_features(
        samples,
        SENSOR_WINDOW_SIZE,
        &features
    );

    print("\n--------------------------------\n");

    switch (scenario) {
    case SENSOR_NORMAL:
        print("MOTOR: HEALTHY SCENARIO (in-distribution)\n");
        break;

    case SENSOR_ABNORMAL_TYPICAL:
        print("MOTOR: ABNORMAL SCENARIO (in-distribution)\n");
        break;

    case SENSOR_ABNORMAL:
    default:
        print("MOTOR: ABNORMAL SCENARIO (SEVERE, out-of-distribution)\n");
        print("  NOTE: features exceed the trained range, so the INT8\n");
        print("  input quantizer saturates and the logit is compressed.\n");
        break;
    }

    print("[FEATURE] Mean absolute: ");
    print_number(features.mean_absolute);
    print(" mg\n");

    print("[FEATURE] RMS: ");
    print_number(features.rms);
    print(" mg\n");

    print("[FEATURE] Peak: ");
    print_number(features.peak);
    print(" mg\n");

    /* Original floating-point model */

    ModelResult float_result = tinyguard_predict(
        (float)features.mean_absolute,
        (float)features.rms,
        (float)features.peak
    );

    /* New INT8 model */

    float input_features[MODEL_INPUTS] = {
        (float)features.mean_absolute,
        (float)features.rms,
        (float)features.peak
    };

    TinyGuardInt8Result int8_result =
        tinyguard_int8_predict(input_features);

    print("\n[FLOAT MODEL] ");

    if (float_result.prediction == 1) {
        print("ANOMALY DETECTED\n");
    } else {
        print("NORMAL OPERATION\n");
    }

    print("[INT8 MODEL] ");

    if (int8_result.prediction == 1) {
        print("ANOMALY DETECTED\n");
    } else {
        print("NORMAL OPERATION\n");
    }

    if (float_result.prediction == int8_result.prediction) {
        print("[VALIDATION] MODEL PREDICTIONS AGREE\n");
    } else {
        print("[VALIDATION] MODEL PREDICTIONS DIFFER\n");
    }
}

int main(void)
{
    print("\n================================\n");
    print("TINYGUARD-AI SENSOR MONITOR\n");
    print("ARM CORTEX-M4 + TINYML\n");
    print("SYNTHETIC DATA / QEMU EMULATION\n");
    print("================================\n");

    print("[BOOT] Firmware initialized\n");
    print("[SENSOR] Simulation started\n");
    print("[AI] FP32 and INT8 models loaded\n");

    monitor_motor(SENSOR_NORMAL);
    monitor_motor(SENSOR_ABNORMAL_TYPICAL);
    monitor_motor(SENSOR_ABNORMAL);

    print("\n[COMPLETE] AI sensor analysis finished\n");

    const uint32_t exit_args[2] = {0x20026u, 0u};

    semihost_call(0x20, exit_args);

    while (1) {
        __asm volatile("wfi");
    }
}
