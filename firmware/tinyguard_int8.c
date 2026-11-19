
#include "tinyguard_int8.h"
#include "model_int8_params.h"

/*
 * Convert a floating-point value to a signed INT8 value.
 * Values are clipped to the symmetric range [-127, 127].
 */
static int8_t quantize_int8(float value, float scale)
{
    float scaled = value / scale;

    int32_t rounded;

    if (scaled >= 0.0f)
        rounded = (int32_t)(scaled + 0.5f);
    else
        rounded = (int32_t)(scaled - 0.5f);

    if (rounded > 127)
        rounded = 127;

    if (rounded < -127)
        rounded = -127;

    return (int8_t)rounded;
}

/*
 * ReLU activation.
 * Negative values become zero.
 */
static int32_t relu_int32(int32_t value)
{
    return (value > 0) ? value : 0;
}

/*
 * TinyGuard-AI INT8 inference engine.
 */
TinyGuardInt8Result tinyguard_int8_predict(
    const float features[MODEL_INPUTS])
{
    TinyGuardInt8Result result;

    int8_t input[MODEL_INPUTS];
    int8_t hidden[MODEL_HIDDEN];

    int32_t acc1[MODEL_HIDDEN];

    /*
     * 1. Normalize sensor features and quantize inputs.
     */
    for (int i = 0; i < MODEL_INPUTS; i++) {

        float normalized =
            (features[i] - scaler_mean[i]) / scaler_std[i];

        input[i] = quantize_int8(
            normalized,
            INPUT_SCALE
        );
    }

    /*
     * 2. First dense layer.
     *
     * Three INT8 inputs multiplied by
     * three sets of eight INT8 weights.
     *
     * Accumulation uses INT32.
     */
    for (int j = 0; j < MODEL_HIDDEN; j++) {

        int32_t sum = bias1[j];

        for (int i = 0; i < MODEL_INPUTS; i++) {

            sum += (int32_t)input[i] *
                   (int32_t)weights1[i][j];
        }

        acc1[j] = relu_int32(sum);
    }

    /*
     * 3. Requantize hidden activations to INT8.
     */
    for (int j = 0; j < MODEL_HIDDEN; j++) {

        float hidden_real =
            (float)acc1[j] * ACC1_SCALE;

        hidden[j] = quantize_int8(
            hidden_real,
            HIDDEN_SCALE
        );

        if (hidden[j] < 0)
            hidden[j] = 0;
    }

    /*
     * 4. Second dense layer.
     */
    int32_t output_sum = bias2[0];

    for (int j = 0; j < MODEL_HIDDEN; j++) {

        output_sum +=
            (int32_t)hidden[j] *
            (int32_t)weights2[j];
    }

    /*
     * 5. Convert output to an approximate logit.
     */
    result.logit =
        (float)output_sum * ACC2_SCALE;

    /*
     * 6. Binary classification.
     */
    result.prediction =
        (output_sum >= 0) ? 1 : 0;

    return result;
}
