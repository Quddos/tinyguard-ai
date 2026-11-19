#include "signal_features.h"

static uint32_t integer_sqrt(uint32_t number)
{
    uint32_t result = 0;
    uint32_t bit = 1u << 30;

    while (bit > number) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (number >= result + bit) {
            number -= result + bit;
            result = (result >> 1) + bit;
        } else {
            result >>= 1;
        }

        bit >>= 2;
    }

    return result;
}

void calculate_features(
    const int32_t *samples,
    uint32_t count,
    SignalFeatures *features)
{
    if (samples == 0 || features == 0 || count == 0) {
        return;
    }

    uint32_t sum_absolute = 0;
    uint32_t sum_squares = 0;
    uint32_t peak = 0;

    for (uint32_t i = 0; i < count; i++) {

        int32_t value = samples[i];

        uint32_t magnitude =
            (value < 0) ? (uint32_t)(-value)
                        : (uint32_t)value;

        sum_absolute += magnitude;
        sum_squares += magnitude * magnitude;

        if (magnitude > peak) {
            peak = magnitude;
        }
    }

    features->mean_absolute = sum_absolute / count;

    features->rms = integer_sqrt(sum_squares / count);

    features->peak = peak;
}
