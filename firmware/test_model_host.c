
#include <stdio.h>
#include "tinyguard_model.h"

int main(void)
{
    ModelResult healthy = tinyguard_predict(11.0f, 11.0f, 16.0f);
    ModelResult abnormal = tinyguard_predict(154.0f, 156.0f, 190.0f);

    printf("HEALTHY INPUT\n");
    printf("Logit: %.9f\n", healthy.logit);
    printf("Prediction: %d\n\n", healthy.prediction);

    printf("ABNORMAL INPUT\n");
    printf("Logit: %.9f\n", abnormal.logit);
    printf("Prediction: %d\n", abnormal.prediction);

    return 0;
}
