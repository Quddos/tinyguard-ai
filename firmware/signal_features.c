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

    /*
     * DEFECT FIX: these accumulators were uint32_t while the input is
     * int32_t. magnitude * magnitude overflows a uint32_t for any
     * |sample| >= 65536, and the running sum overflows sooner still, which
     * silently produced a wrong RMS instead of failing. Current synthetic
     * windows peak near 190 so the bug was unreachable in practice, but the
     * public API accepts int32_t samples, so accumulate in uint64_t.
     */
    uint64_t sum_absolute = 0;
    uint64_t sum_squares = 0;
    uint32_t peak = 0;

    for (uint32_t i = 0; i < count; i++) {

        int32_t value = samples[i];

        /* Negating INT32_MIN is undefined; cast first, then negate. */
        uint32_t magnitude =
            (value < 0) ? (uint32_t)(-(int64_t)value)
                        : (uint32_t)value;

        sum_absolute += (uint64_t)magnitude;
        sum_squares += (uint64_t)magnitude * (uint64_t)magnitude;

        if (magnitude > peak) {
            peak = magnitude;
        }
    }

    features->mean_absolute = (uint32_t)(sum_absolute / count);

    /*
     * integer_sqrt takes uint32_t. mean_square cannot exceed UINT32_MAX for
     * any physically meaningful window, but clamp rather than truncate so an
     * extreme input degrades predictably instead of wrapping.
     */
    uint64_t mean_square = sum_squares / count;

    if (mean_square > 0xFFFFFFFFu) {
        mean_square = 0xFFFFFFFFu;
    }

    features->rms = integer_sqrt((uint32_t)mean_square);

    features->peak = peak;
}
