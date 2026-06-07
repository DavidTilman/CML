#include "tensor.h"

#include <stdio.h>
#include <stdlib.h>

#define MAX_DIMS 32

// --- Lifecycle ---------------------------------------------------------------

Tensor *tensor_create(size_t *shape, size_t ndim) {
    Tensor *t  = malloc(sizeof(Tensor));
    t->ndim    = ndim;
    t->shape   = malloc(ndim * sizeof(size_t));
    t->strides = malloc(ndim * sizeof(size_t));
    t->size    = 1;
    for (size_t i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->size    *= shape[i];
    }
    t->strides[ndim - 1] = 1;
    for (size_t i = ndim - 1; i-- > 0; )
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    t->data = calloc(t->size, sizeof(float));
    return t;
}

void tensor_free(Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

// --- Access ------------------------------------------------------------------

float tensor_get(Tensor *t, size_t *idx) {
    size_t flat = 0;
    for (size_t i = 0; i < t->ndim; i++)
        flat += idx[i] * t->strides[i];
    return t->data[flat];
}

void tensor_set(Tensor *t, size_t *idx, float val) {
    size_t flat = 0;
    for (size_t i = 0; i < t->ndim; i++)
        flat += idx[i] * t->strides[i];
    t->data[flat] = val;
}

int tensors_comparable(Tensor *a, Tensor *b) {
    if (a->ndim != b->ndim) return 0;
    for (size_t i = 0; i < a->ndim; i++)
        if (a->shape[i] != b->shape[i]) return 0;
    return a->size == b->size;
}

// --- Shape operations --------------------------------------------------------

Tensor *tensor_reshape(Tensor *t, size_t *new_shape, size_t new_ndim) {
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) new_size *= new_shape[i];
    if (new_size != t->size) return NULL;

    Tensor *out = tensor_create(new_shape, new_ndim);
    for (size_t i = 0; i < t->size; i++) out->data[i] = t->data[i];
    return out;
}

void tensor_reshape_(Tensor *t, size_t *new_shape, size_t new_ndim) {
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) new_size *= new_shape[i];
    if (new_size != t->size) return;

    t->shape   = realloc(t->shape,   new_ndim * sizeof(size_t));
    t->strides = realloc(t->strides, new_ndim * sizeof(size_t));
    t->ndim    = new_ndim;
    for (size_t i = 0; i < new_ndim; i++) t->shape[i] = new_shape[i];
    t->strides[new_ndim - 1] = 1;
    for (size_t i = new_ndim - 1; i-- > 0; )
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
}

Tensor *tensor_transpose(Tensor *t) {
    Tensor *out = tensor_create(t->shape, t->ndim);
    for (size_t i = 0; i < t->ndim; i++)
        out->shape[i] = t->shape[t->ndim - 1 - i];
    out->strides[out->ndim - 1] = 1;
    for (size_t i = out->ndim - 1; i-- > 0; )
        out->strides[i] = out->strides[i + 1] * out->shape[i + 1];

    size_t idx[MAX_DIMS];
    for (size_t i = 0; i < t->size; i++) {
        size_t tmp = i;
        for (size_t d = out->ndim; d-- > 0; ) {
            idx[d] = tmp % out->shape[d];
            tmp   /= out->shape[d];
        }
        size_t src = 0;
        for (size_t d = 0; d < t->ndim; d++)
            src += idx[d] * t->strides[t->ndim - 1 - d];
        out->data[i] = t->data[src];
    }
    return out;
}

// --- Utility -----------------------------------------------------------------

void tensor_print(Tensor *t) {
    printf("<\n\tTensor at: %p\n\t", t);
    printf("shape:[");
    for (size_t i = 0; i < t->ndim; i++)
        printf("%lu, ", t->shape[i]);
    printf("]\n\t");
    printf("size: %lu\n\t", t->size);
    printf("sum: %f\n\t", tensor_sum(t));
    printf("mean: %f\n\t", tensor_mean(t));
    printf("max: %f\n\t", tensor_max(t));
    printf("min: %f\n", tensor_min(t));
    printf(">\n");
}

void tensor_fill(Tensor *t, float_t value) {
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = value;
}

void tensor_zeros(Tensor *t) {
    tensor_fill(t, 0);
}

void tensor_ones(Tensor *t) {
    tensor_fill(t, 1);
}

Tensor *tensor_copy(Tensor *t) {
    Tensor *out = tensor_create(t->shape, t->ndim);
    for (size_t i = 0; i < t->size; i++) out->data[i] = t->data[i];
    return out;
}
