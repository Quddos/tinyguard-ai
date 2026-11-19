
#include <stdio.h>
#include <stdint.h>
#include "model_int8_params.h"

static int8_t q8(float value, float scale)
{
    float v = value / scale;
    int32_t q = (v >= 0.0f)
        ? (int32_t)(v + 0.5f)
        : (int32_t)(v - 0.5f);

    if (q > 127) q = 127;
    if (q < -127) q = -127;

    return (int8_t)q;
}

int main(void)
{
    float raw[3];
    float normalized[3];
    int8_t input[3];
    int32_t acc1[8];
    int8_t hidden[8];

    if (scanf("%f %f %f", &raw[0], &raw[1], &raw[2]) != 3)
        return 1;

    printf("RAW: ");

    for (int i = 0; i < 3; i++)
        printf("%.6f ", raw[i]);

    printf("\nNORMALIZED: ");

    for (int i = 0; i < 3; i++) {
        normalized[i] =
            (raw[i] - scaler_mean[i]) / scaler_std[i];

        printf("%.6f ", normalized[i]);

        input[i] = q8(normalized[i], INPUT_SCALE);
    }

    printf("\nINPUT INT8: ");

    for (int i = 0; i < 3; i++)
        printf("%d ", input[i]);

    printf("\nACC1: ");

    for (int j = 0; j < 8; j++) {
        int32_t sum = bias1[j];

        for (int i = 0; i < 3; i++)
            sum += (int32_t)input[i] * weights1[i][j];

        acc1[j] = sum > 0 ? sum : 0;

        printf("%ld ", (long)acc1[j]);
    }

    printf("\nHIDDEN INT8: ");

    for (int j = 0; j < 8; j++) {
        float real = (float)acc1[j] * ACC1_SCALE;

        hidden[j] = q8(real, HIDDEN_SCALE);

        if (hidden[j] < 0)
            hidden[j] = 0;

        printf("%d ", hidden[j]);
    }

    int32_t output = bias2[0];

    for (int j = 0; j < 8; j++)
        output += (int32_t)hidden[j] * weights2[j];

    printf("\nOUTPUT SUM: %ld", (long)output);
    printf("\nLOGIT: %.9f\n", (float)output * ACC2_SCALE);

    return 0;
}
