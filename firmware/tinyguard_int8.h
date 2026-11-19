
#ifndef TINYGUARD_INT8_H
#define TINYGUARD_INT8_H

#include <stdint.h>

typedef struct {
    float logit;
    int prediction;
} TinyGuardInt8Result;

/*
 * Run quantized neural-network inference.
 *
 * Input:
 *   features[0] = mean absolute
 *   features[1] = RMS
 *   features[2] = peak
 *
 * Output:
 *   prediction 0 = Normal
 *   prediction 1 = Anomaly
 */
TinyGuardInt8Result tinyguard_int8_predict(
    const float features[3]
);

#endif
