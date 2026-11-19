#ifndef TINYGUARD_MODEL_PARAMS_H
#define TINYGUARD_MODEL_PARAMS_H

/* Automatically exported from the trained Python MLP.
   Input order: mean_absolute, rms, peak.
   Hidden activation: ReLU.
   Output: binary classification logit.
*/

#define MODEL_INPUTS 3
#define MODEL_HIDDEN 8

static const float scaler_mean[MODEL_INPUTS] = {
    17.0904167f, 21.8191667f, 45.115f
};

static const float scaler_scale[MODEL_INPUTS] = {
    15.3997481f, 18.2693678f, 37.3224701f
};

static const float weights1[MODEL_INPUTS][MODEL_HIDDEN] = {
    {-1.21757001f, 1.87814649f, 0.444289161f, 0.674213471f, -0.568697562f, -1.73868725f, -0.864150965f, 0.540619173f},
    {-1.12561511f, 1.97660418f, -0.594371063f, 1.36783627f, 0.374951585f, -1.94786122f, -0.757904266f, -0.467418738f},
    {-1.59496627f, 1.93218312f, 0.014834777f, 0.361096486f, -0.0421251874f, -2.05929898f, -0.571916466f, -0.197302174f}
};

static const float bias1[MODEL_HIDDEN] = {
    -0.440552904f, 3.36534251f, -0.565079603f, 1.43416f, -0.139396753f, -1.21847306f, -0.434260411f, -0.486668122f
};

static const float weights2[MODEL_HIDDEN] = {
    -2.29036158f, 4.67784783f, 0.755830568f, 2.10717241f, -0.00697526026f, -3.27510805f, -1.45909615f, -0.097683799f
};

static const float bias2 = -0.0721872127f;

#endif
