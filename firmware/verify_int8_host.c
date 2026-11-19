
#include <stdio.h>
#include "tinyguard_int8.h"

int main(void)
{
    float features[3];

    while (scanf("%f %f %f",
                 &features[0],
                 &features[1],
                 &features[2]) == 3)
    {
        TinyGuardInt8Result result =
            tinyguard_int8_predict(features);

        printf("%.9f %d\n",
               result.logit,
               result.prediction);
    }

    return 0;
}
