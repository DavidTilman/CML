#include "activation.h"
#include <math.h>

float_t act_relu(float_t x)     { return x > 0.0f ? x : 0.0f; }
float_t act_logistic(float_t x) { return 1.0f / (1.0f + expf(-x)); }
float_t act_tanh(float_t x)     { return tanhf(x); }
float_t act_exp(float_t x)      { return expf(x); }
float_t act_log(float_t x)      { return logf(x); }
float_t act_sqrt(float_t x)     { return sqrtf(x); }
