#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <math.h>
#include "activation.h"

typedef struct {
    float  *data;
    size_t *shape;
    size_t *strides;
    size_t  ndim;
    size_t  size;
} Tensor;

// Lifecycle
Tensor  *tensor_create(size_t *shape, size_t ndim);
void     tensor_free(Tensor *t);

// Access
float    tensor_get(Tensor *t, size_t *idx);
void     tensor_set(Tensor *t, size_t *idx, float val);
int      tensors_comparable(Tensor *a, Tensor *b);

// Shape operations
Tensor  *tensor_reshape(Tensor *t, size_t *new_shape, size_t new_ndim);
void     tensor_reshape_(Tensor *t, size_t *new_shape, size_t new_ndim);
Tensor  *tensor_transpose(Tensor *t);

// Element-wise arithmetic
Tensor  *tensor_add(Tensor *a, Tensor *b);
Tensor  *tensor_add_s(Tensor *a, float_t s);
void     tensor_add_s_(Tensor *a, float_t s);

Tensor  *tensor_sub(Tensor *a, Tensor *b);
Tensor  *tensor_sub_s(Tensor *a, float_t s);
void     tensor_sub_s_(Tensor *a, float_t s);

Tensor  *tensor_mult(Tensor *a, Tensor *b);
void     tensor_mult_(Tensor *a, Tensor *b);

Tensor  *tensor_scale(Tensor *a, float_t s);
void     tensor_scale_(Tensor *a, float_t s);

Tensor  *tensor_div(Tensor *a, Tensor *b);
void     tensor_div_(Tensor *a, Tensor *b);

Tensor  *tensor_pow(Tensor *t, float_t e);
void     tensor_pow_(Tensor *t, float_t e);

// Linear algebra
Tensor  *tensor_matmul(Tensor *a, Tensor *b);

// Reductions (global)
float_t  tensor_sum(Tensor *t);
float_t  tensor_mean(Tensor *t);
float_t  tensor_max(Tensor *t);
float_t  tensor_min(Tensor *t);

// Reductions (axis)
Tensor  *tensor_sum_axis(Tensor *t, size_t axis);
Tensor  *tensor_mean_axis(Tensor *t, size_t axis);
Tensor  *tensor_max_axis(Tensor *t, size_t axis);
Tensor  *tensor_min_axis(Tensor *t, size_t axis);

// Activation
Tensor  *tensor_activate(Tensor *t, float_t (*fn)(float_t));
void     tensor_activate_(Tensor *t, float_t (*fn)(float_t));

Tensor  *tensor_relu(Tensor *t);
Tensor  *tensor_logistic(Tensor *t);
Tensor  *tensor_tanh(Tensor *t);
Tensor  *tensor_exp(Tensor *t);
Tensor  *tensor_log(Tensor *t);
Tensor  *tensor_sqrt(Tensor *t);

void     tensor_relu_(Tensor *t);
void     tensor_logistic_(Tensor *t);
void     tensor_tanh_(Tensor *t);
void     tensor_exp_(Tensor *t);
void     tensor_log_(Tensor *t);
void     tensor_sqrt_(Tensor *t);

// Softmax
Tensor  *tensor_softmax(Tensor *t);
void     tensor_softmax_(Tensor *t);
Tensor  *tensor_softmax_axis(Tensor *t, size_t axis);
void     tensor_softmax_axis_(Tensor *t, size_t axis);

// Random
void     tensor_seed(unsigned int seed);
void     tensor_rand_range(Tensor *t, float_t min, float_t max);

// Utility
Tensor  *tensor_copy(Tensor *t);
void     tensor_fill(Tensor *t, float_t value);
void     tensor_zeros(Tensor *t);
void     tensor_ones(Tensor *t);
void     tensor_print(Tensor *t);

#endif
