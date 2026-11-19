#ifndef TINYGUARD_MODEL_H
#define TINYGUARD_MODEL_H

typedef struct
{
    float logit;
    int prediction;
} ModelResult;

ModelResult tinyguard_predict(
    float mean_absolute,
    float rms,
    float peak
);

#endif
