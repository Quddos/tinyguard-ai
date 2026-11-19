#ifndef MODEL_INT8_PARAMS_H
#define MODEL_INT8_PARAMS_H

#include <stdint.h>

#define MODEL_INPUTS 3
#define MODEL_HIDDEN 8

#define INPUT_SCALE 0.0311435647309f
#define W1_SCALE 0.0162149525632f
#define ACC1_SCALE 0.000504991424759f
#define HIDDEN_SCALE 0.177796744619f
#define W2_SCALE 0.036833447484f
#define ACC2_SCALE 0.00654886705577f

static const float scaler_mean[3] = {
    17.0904166667f, 21.8191666667f, 45.115f
};

static const float scaler_std[3] = {
    15.3997480984f, 18.2693677861f, 37.3224701085f
};

static const int8_t weights1[3][8] = {
    {-75, 116, 27, 42, -35, -107, -53, 33},
    {-69, 122, -37, 84, 23, -120, -47, -29},
    {-98, 119, 1, 22, -3, -127, -35, -12},
};

static const int32_t bias1[8] = {
    -872, 6664, -1119, 2840, -276, -2413, -860, -964
};

static const int8_t weights2[8] = {
    -62, 127, 21, 57, 0, -89, -40, -3
};

static const int32_t bias2[1] = {
    -11
};

#endif
