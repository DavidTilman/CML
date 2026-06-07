#include "tensor.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define MAX_DIMS 32

// --- Activation --------------------------------------------------------------

Tensor *tensor_activate(Tensor *t, float_t (*fn)(float_t)) {
    Tensor *out = tensor_create(t->shape, t->ndim);
    for (size_t i = 0; i < t->size; i++) out->data[i] = fn(t->data[i]);
    return out;
}

void tensor_activate_(Tensor *t, float_t (*fn)(float_t)) {
    for (size_t i = 0; i < t->size; i++) t->data[i] = fn(t->data[i]);
}

Tensor *tensor_relu(Tensor *t)     { return tensor_activate(t, act_relu); }
Tensor *tensor_logistic(Tensor *t) { return tensor_activate(t, act_logistic); }
Tensor *tensor_tanh(Tensor *t)     { return tensor_activate(t, act_tanh); }
Tensor *tensor_exp(Tensor *t)      { return tensor_activate(t, act_exp); }
Tensor *tensor_log(Tensor *t)      { return tensor_activate(t, act_log); }
Tensor *tensor_sqrt(Tensor *t)     { return tensor_activate(t, act_sqrt); }

void tensor_relu_(Tensor *t)     { tensor_activate_(t, act_relu); }
void tensor_logistic_(Tensor *t) { tensor_activate_(t, act_logistic); }
void tensor_tanh_(Tensor *t)     { tensor_activate_(t, act_tanh); }
void tensor_exp_(Tensor *t)      { tensor_activate_(t, act_exp); }
void tensor_log_(Tensor *t)      { tensor_activate_(t, act_log); }
void tensor_sqrt_(Tensor *t)     { tensor_activate_(t, act_sqrt); }

// --- Softmax -----------------------------------------------------------------

Tensor *tensor_softmax(Tensor *t) {
    Tensor *out = tensor_create(t->shape, t->ndim);
    float max = tensor_max(t);
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        out->data[i] = expf(t->data[i] - max);
        sum += out->data[i];
    }
    for (size_t i = 0; i < t->size; i++) out->data[i] /= sum;
    return out;
}

void tensor_softmax_(Tensor *t) {
    float max = tensor_max(t);
    float sum = 0.0f;
    for (size_t i = 0; i < t->size; i++) {
        t->data[i] = expf(t->data[i] - max);
        sum += t->data[i];
    }
    for (size_t i = 0; i < t->size; i++) t->data[i] /= sum;
}

static void softmax_axis_inplace(Tensor *t, size_t axis) {
    size_t axis_len   = t->shape[axis];
    size_t outer_size = t->size / axis_len;

    size_t outer_shape[MAX_DIMS];
    size_t outer_ndim = t->ndim - 1;
    for (size_t i = 0, j = 0; i < t->ndim; i++)
        if (i != axis) outer_shape[j++] = t->shape[i];

    size_t outer_strides[MAX_DIMS];
    if (outer_ndim > 0) {
        outer_strides[outer_ndim - 1] = 1;
        for (size_t i = outer_ndim - 1; i-- > 0; )
            outer_strides[i] = outer_strides[i + 1] * outer_shape[i + 1];
    }

    for (size_t s = 0; s < outer_size; s++) {
        size_t base = 0;
        size_t tmp  = s;
        size_t outer_idx[MAX_DIMS];
        for (size_t d = outer_ndim; d-- > 0; ) {
            outer_idx[d] = tmp % outer_shape[d];
            tmp          /= outer_shape[d];
        }
        size_t j = 0;
        for (size_t d = 0; d < t->ndim; d++)
            if (d != axis) base += outer_idx[j++] * t->strides[d];

        float max_val = -FLT_MAX;
        for (size_t k = 0; k < axis_len; k++) {
            float v = t->data[base + k * t->strides[axis]];
            if (v > max_val) max_val = v;
        }

        float sum = 0.0f;
        for (size_t k = 0; k < axis_len; k++) {
            float e = expf(t->data[base + k * t->strides[axis]] - max_val);
            t->data[base + k * t->strides[axis]] = e;
            sum += e;
        }

        for (size_t k = 0; k < axis_len; k++)
            t->data[base + k * t->strides[axis]] /= sum;
    }
}

Tensor *tensor_softmax_axis(Tensor *t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    Tensor *out = tensor_copy(t);
    softmax_axis_inplace(out, axis);
    return out;
}

void tensor_softmax_axis_(Tensor *t, size_t axis) {
    if (axis >= t->ndim) return;
    softmax_axis_inplace(t, axis);
}

// --- Random ------------------------------------------------------------------

void tensor_seed(unsigned int seed) { srand(seed); }

void tensor_rand_range(Tensor *t, float_t min, float_t max) {
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = min + ((float_t)rand() / RAND_MAX) * (max - min);
}
