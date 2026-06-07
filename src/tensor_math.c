#include "tensor.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>

#define MAX_DIMS 32

// --- Internal helpers --------------------------------------------------------

static int broadcast_shapes(
    size_t *sa, size_t ndim_a,
    size_t *sb, size_t ndim_b,
    size_t *out_shape, size_t *out_ndim)
{
    size_t ndim = ndim_a > ndim_b ? ndim_a : ndim_b;
    *out_ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        size_t da = (i < ndim - ndim_a) ? 1 : sa[i - (ndim - ndim_a)];
        size_t db = (i < ndim - ndim_b) ? 1 : sb[i - (ndim - ndim_b)];
        if (da != db && da != 1 && db != 1) return 0;
        out_shape[i] = da > db ? da : db;
    }
    return 1;
}

static size_t broadcast_flat_idx(
    Tensor *t, size_t *out_strides, size_t *out_shape, size_t out_ndim, size_t flat)
{
    size_t offset = out_ndim - t->ndim;
    size_t result = 0;
    for (size_t i = 0; i < t->ndim; i++) {
        size_t out_i = i + offset;
        size_t idx = (flat / out_strides[out_i]) % out_shape[out_i];
        if (t->shape[i] != 1)
            result += idx * t->strides[i];
    }
    return result;
}

static float fn_add(float a, float b) { return a + b; }
static float fn_max(float a, float b) { return a > b ? a : b; }
static float fn_min(float a, float b) { return a < b ? a : b; }

static Tensor *reduce_axis(Tensor *t, size_t axis, float init, float (*fn)(float, float)) {
    if (axis >= t->ndim) return NULL;

    size_t out_shape[MAX_DIMS];
    size_t out_ndim = t->ndim - 1;
    for (size_t i = 0, j = 0; i < t->ndim; i++)
        if (i != axis) out_shape[j++] = t->shape[i];

    Tensor *out = tensor_create(out_shape, out_ndim);
    for (size_t i = 0; i < out->size; i++) out->data[i] = init;

    for (size_t flat = 0; flat < out->size; flat++) {
        size_t out_idx[MAX_DIMS];
        size_t tmp = flat;
        for (size_t d = out_ndim; d-- > 0; ) {
            out_idx[d] = tmp % out_shape[d];
            tmp       /= out_shape[d];
        }
        size_t in_idx[MAX_DIMS];
        for (size_t k = 0; k < t->shape[axis]; k++) {
            size_t j = 0;
            for (size_t d = 0; d < t->ndim; d++) {
                if (d == axis) in_idx[d] = k;
                else           in_idx[d] = out_idx[j++];
            }
            size_t src = 0;
            for (size_t d = 0; d < t->ndim; d++)
                src += in_idx[d] * t->strides[d];
            out->data[flat] = fn(out->data[flat], t->data[src]);
        }
    }
    return out;
}

// --- Element-wise arithmetic -------------------------------------------------

Tensor *tensor_add(Tensor *a, Tensor *b) {
    size_t out_shape[MAX_DIMS], out_ndim;
    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim))
        return NULL;
    Tensor *out = tensor_create(out_shape, out_ndim);
    for (size_t flat = 0; flat < out->size; flat++)
        out->data[flat] = a->data[broadcast_flat_idx(a, out->strides, out_shape, out_ndim, flat)]
                        + b->data[broadcast_flat_idx(b, out->strides, out_shape, out_ndim, flat)];
    return out;
}

Tensor *tensor_add_s(Tensor *a, float_t s) {
    Tensor *out = tensor_create(a->shape, a->ndim);
    for (size_t i = 0; i < out->size; i++) out->data[i] = a->data[i] + s;
    return out;
}

void tensor_add_s_(Tensor *a, float_t s) {
    for (size_t i = 0; i < a->size; i++) a->data[i] += s;
}

Tensor *tensor_sub(Tensor *a, Tensor *b) {
    size_t out_shape[MAX_DIMS], out_ndim;
    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim))
        return NULL;
    Tensor *out = tensor_create(out_shape, out_ndim);
    for (size_t flat = 0; flat < out->size; flat++)
        out->data[flat] = a->data[broadcast_flat_idx(a, out->strides, out_shape, out_ndim, flat)]
                        - b->data[broadcast_flat_idx(b, out->strides, out_shape, out_ndim, flat)];
    return out;
}

Tensor *tensor_sub_s(Tensor *a, float_t s) {
    Tensor *out = tensor_create(a->shape, a->ndim);
    for (size_t i = 0; i < out->size; i++) out->data[i] = a->data[i] - s;
    return out;
}

void tensor_sub_s_(Tensor *a, float_t s) {
    for (size_t i = 0; i < a->size; i++) a->data[i] -= s;
}

Tensor *tensor_mult(Tensor *a, Tensor *b) {
    size_t out_shape[MAX_DIMS], out_ndim;
    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim))
        return NULL;
    Tensor *out = tensor_create(out_shape, out_ndim);
    for (size_t flat = 0; flat < out->size; flat++)
        out->data[flat] = a->data[broadcast_flat_idx(a, out->strides, out_shape, out_ndim, flat)]
                        * b->data[broadcast_flat_idx(b, out->strides, out_shape, out_ndim, flat)];
    return out;
}

void tensor_mult_(Tensor *a, Tensor *b) {
    for (size_t i = 0; i < a->size; i++) a->data[i] *= b->data[i];
}

Tensor *tensor_scale(Tensor *a, float_t s) {
    Tensor *out = tensor_create(a->shape, a->ndim);
    for (size_t i = 0; i < out->size; i++) out->data[i] = a->data[i] * s;
    return out;
}

void tensor_scale_(Tensor *a, float_t s) {
    for (size_t i = 0; i < a->size; i++) a->data[i] *= s;
}

Tensor *tensor_div(Tensor *a, Tensor *b) {
    size_t out_shape[MAX_DIMS], out_ndim;
    if (!broadcast_shapes(a->shape, a->ndim, b->shape, b->ndim, out_shape, &out_ndim))
        return NULL;
    Tensor *out = tensor_create(out_shape, out_ndim);
    for (size_t flat = 0; flat < out->size; flat++)
        out->data[flat] = a->data[broadcast_flat_idx(a, out->strides, out_shape, out_ndim, flat)]
                        / b->data[broadcast_flat_idx(b, out->strides, out_shape, out_ndim, flat)];
    return out;
}

void tensor_div_(Tensor *a, Tensor *b) {
    for (size_t i = 0; i < a->size; i++) a->data[i] /= b->data[i];
}

Tensor *tensor_pow(Tensor *b, float_t e) {
    Tensor *t = tensor_create(b->shape, b->ndim);
    for (size_t i = 0; i < b->size; i++)
        t->data[i] = powf(b->data[i], e);
    return t;
}

void tensor_pow_(Tensor *b, float_t e) {
    for (size_t i = 0; i < b->size; i++)
        b->data[i] = powf(b->data[i], e);
}

// --- Linear algebra ----------------------------------------------------------

Tensor *tensor_matmul(Tensor *a, Tensor *b) {
    if (a->ndim < 2 || b->ndim < 2) return NULL;

    size_t M  = a->shape[a->ndim - 2];
    size_t K  = a->shape[a->ndim - 1];
    size_t K2 = b->shape[b->ndim - 2];
    size_t N  = b->shape[b->ndim - 1];
    if (K != K2) return NULL;

    size_t batch_ndim_a = a->ndim - 2;
    size_t batch_ndim_b = b->ndim - 2;

    size_t batch_shape[MAX_DIMS];
    size_t batch_ndim = 0;
    if (batch_ndim_a > 0 || batch_ndim_b > 0) {
        if (!broadcast_shapes(a->shape, batch_ndim_a, b->shape, batch_ndim_b,
                              batch_shape, &batch_ndim))
            return NULL;
    }

    size_t out_shape[MAX_DIMS];
    for (size_t i = 0; i < batch_ndim; i++) out_shape[i] = batch_shape[i];
    out_shape[batch_ndim]     = M;
    out_shape[batch_ndim + 1] = N;
    Tensor *out = tensor_create(out_shape, batch_ndim + 2);

    size_t batch_strides[MAX_DIMS];
    if (batch_ndim > 0) {
        batch_strides[batch_ndim - 1] = 1;
        for (size_t i = batch_ndim - 1; i-- > 0; )
            batch_strides[i] = batch_strides[i + 1] * batch_shape[i + 1];
    }

    size_t batch_size = out->size / (M * N);
    size_t off_a = batch_ndim - batch_ndim_a;
    size_t off_b = batch_ndim - batch_ndim_b;

    for (size_t bf = 0; bf < batch_size; bf++) {
        size_t a_base = 0, b_base = 0;
        for (size_t i = 0; i < batch_ndim; i++) {
            size_t idx = (bf / batch_strides[i]) % batch_shape[i];
            if (i >= off_a && a->shape[i - off_a] != 1)
                a_base += idx * a->strides[i - off_a];
            if (i >= off_b && b->shape[i - off_b] != 1)
                b_base += idx * b->strides[i - off_b];
        }
        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < N; j++)
                for (size_t k = 0; k < K; k++)
                    out->data[bf*M*N + i*N + j] +=
                        a->data[a_base + i*a->strides[a->ndim-2] + k*a->strides[a->ndim-1]] *
                        b->data[b_base + k*b->strides[b->ndim-2] + j*b->strides[b->ndim-1]];
    }
    return out;
}

// --- Reductions (global) -----------------------------------------------------

float_t tensor_sum(Tensor *t) {
    float_t s = 0;
    for (size_t i = 0; i < t->size; i++) s += t->data[i];
    return s;
}

float_t tensor_mean(Tensor *t) {
    return tensor_sum(t) / t->size;
}

float_t tensor_max(Tensor *t) {
    float_t m = -FLT_MAX;
    for (size_t i = 0; i < t->size; i++)
        if (t->data[i] > m) m = t->data[i];
    return m;
}

float_t tensor_min(Tensor *t) {
    float_t m = FLT_MAX;
    for (size_t i = 0; i < t->size; i++)
        if (t->data[i] < m) m = t->data[i];
    return m;
}

// --- Reductions (axis) -------------------------------------------------------

Tensor *tensor_sum_axis(Tensor *t, size_t axis) {
    return reduce_axis(t, axis, 0.0f, fn_add);
}

Tensor *tensor_mean_axis(Tensor *t, size_t axis) {
    if (axis >= t->ndim) return NULL;
    Tensor *out = reduce_axis(t, axis, 0.0f, fn_add);
    float n = (float)t->shape[axis];
    for (size_t i = 0; i < out->size; i++) out->data[i] /= n;
    return out;
}

Tensor *tensor_max_axis(Tensor *t, size_t axis) {
    return reduce_axis(t, axis, -FLT_MAX, fn_max);
}

Tensor *tensor_min_axis(Tensor *t, size_t axis) {
    return reduce_axis(t, axis, FLT_MAX, fn_min);
}
