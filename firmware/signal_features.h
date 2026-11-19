#ifndef SIGNAL_FEATURES_H
#define SIGNAL_FEATURES_H

#include <stdint.h>

typedef struct {
    uint32_t mean_absolute;
    uint32_t rms;
    uint32_t peak;
} SignalFeatures;

void calculate_features(
    const int32_t *samples,
    uint32_t count,
    SignalFeatures *features
);

#endif
