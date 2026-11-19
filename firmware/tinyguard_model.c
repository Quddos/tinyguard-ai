
#include "tinyguard_model.h"
#include "model_params.h"

ModelResult tinyguard_predict(
    float mean_absolute,
    float rms,
    float peak
)
{
    float input[MODEL_INPUTS];
    float hidden[MODEL_HIDDEN];

    float raw_features[MODEL_INPUTS] = {
        mean_absolute,
        rms,
        peak
    };

    /* Step 1: Normalize input features */
    for (int i = 0; i < MODEL_INPUTS; i++)
    {
        input[i] =
            (raw_features[i] - scaler_mean[i]) /
            scaler_scale[i];
    }

    /* Step 2: Hidden layer calculation */
    for (int j = 0; j < MODEL_HIDDEN; j++)
    {
        float sum = bias1[j];

        for (int i = 0; i < MODEL_INPUTS; i++)
        {
            sum += input[i] * weights1[i][j];
        }

        /* ReLU activation */
        hidden[j] = (sum > 0.0f) ? sum : 0.0f;
    }

    /* Step 3: Output layer */
    float logit = bias2;

    for (int j = 0; j < MODEL_HIDDEN; j++)
    {
        logit += hidden[j] * weights2[j];
    }

    /* Step 4: Binary classification */
    ModelResult result;

    result.logit = logit;
    result.prediction = (logit >= 0.0f) ? 1 : 0;

    return result;
}
