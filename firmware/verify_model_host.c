
#include <stdio.h>
#include "tinyguard_model.h"

int main(void)
{
    float mean_absolute;
    float rms;
    float peak;

    while (scanf("%f,%f,%f",
                 &mean_absolute,
                 &rms,
                 &peak) == 3)
    {
        ModelResult result = tinyguard_predict(
            mean_absolute,
            rms,
            peak
        );

        printf("%.9g,%d\n",
               result.logit,
               result.prediction);
    }

    return 0;
}
