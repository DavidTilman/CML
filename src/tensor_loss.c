#include "tensor.h"

#include <float.h>

float_t tensor_mse_loss(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    float_t acc = 0;
    for (size_t i = 0; i < a->size; i++) {
        float_t r = a->data[i] - b->data[i];
        acc += r * r;
    }
    return acc / a->size;
}

float_t tensor_mae_loss(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    float_t acc = 0;
    for (size_t i = 0; i < a->size; i++) {
        acc += fabsf(a->data[i] - b->data[i]);
    }
    return acc / a->size;
}

float_t tensor_huber_loss(Tensor *a, Tensor *b, float_t delta) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    float_t acc = 0;
    for (size_t i = 0; i < a->size; i++) {
        float_t r = fabsf(a->data[i] - b->data[i]);
        acc += (r <= delta) ? 0.5f * r * r : delta * (r - 0.5f * delta);
    }
    return acc / a->size;
}

float_t tensor_bce_loss(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    static const float_t eps = 1e-7f;
    float_t acc = 0;
    for (size_t i = 0; i < a->size; i++) {
        float_t p = a->data[i] < eps ? eps : (a->data[i] > 1.0f - eps ? 1.0f - eps : a->data[i]);
        acc += b->data[i] * logf(p) + (1.0f - b->data[i]) * logf(1.0f - p);
    }
    return -acc / a->size;
}

float_t tensor_cce_loss(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    size_t N = a->shape[0];
    size_t C = a->size / N;
    float_t total = 0;
    for (size_t i = 0; i < N; i++) {
        float_t sample = 0;
        for (size_t j = 0; j < C; j++)
            sample += b->data[i * C + j] * logf(a->data[i * C + j]);
        total += -sample;
    }
    return total / N;
}

float_t tensor_hinge_loss(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return FLT_MAX;

    size_t N = a->shape[0];
    float_t acc = 0;
    for (size_t i = 0; i < a->size; i++) {
        float_t h = 1.0f - b->data[i] * a->data[i];
        acc += h > 0.0f ? h : 0.0f;
    }
    return acc / N;
}

Tensor *tensor_mse_grad(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return NULL;

    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t n = (float_t)a->size;
    for (size_t i = 0; i < a->size; i++)
        grad->data[i] = 2.0f * (a->data[i] - b->data[i]) / n;
    return grad;
}

Tensor *tensor_mae_grad(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return NULL;

    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t n = (float_t)a->size;
    for (size_t i = 0; i < a->size; i++) {
        float_t r = a->data[i] - b->data[i];
        grad->data[i] = (r > 0.0f ? 1.0f : (r < 0.0f ? -1.0f : 0.0f)) / n;
    }
    return grad;
}

Tensor *tensor_huber_grad(Tensor *a, Tensor *b, float_t delta) {
    if (!tensors_comparable(a, b)) return NULL;

    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t n = (float_t)a->size;
    for (size_t i = 0; i < a->size; i++) {
        float_t r = a->data[i] - b->data[i];
        grad->data[i] = (fabsf(r) <= delta ? r : delta * (r > 0.0f ? 1.0f : -1.0f)) / n;
    }
    return grad;
}

Tensor *tensor_bce_grad(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return NULL;

    static const float_t eps = 1e-7f;
    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t n = (float_t)a->size;
    for (size_t i = 0; i < a->size; i++) {
        float_t p = a->data[i] < eps ? eps : (a->data[i] > 1.0f - eps ? 1.0f - eps : a->data[i]);
        grad->data[i] = (p - b->data[i]) / (p * (1.0f - p) * n);
    }
    return grad;
}

Tensor *tensor_cce_grad(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return NULL;

    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t N = (float_t)a->shape[0];
    for (size_t i = 0; i < a->size; i++)
        grad->data[i] = -b->data[i] / (a->data[i] * N);
    return grad;
}

Tensor *tensor_hinge_grad(Tensor *a, Tensor *b) {
    if (!tensors_comparable(a, b)) return NULL;

    Tensor *grad = tensor_create(a->shape, a->ndim);
    float_t N = (float_t)a->shape[0];
    for (size_t i = 0; i < a->size; i++)
        grad->data[i] = (1.0f - b->data[i] * a->data[i] > 0.0f) ? -b->data[i] / N : 0.0f;
    return grad;
}

Tensor *tensor_kl_grad(Tensor *p, Tensor *q) {
    if (!tensors_comparable(p, q)) return NULL;

    Tensor *grad = tensor_create(q->shape, q->ndim);
    for (size_t i = 0; i < q->size; i++)
        grad->data[i] = p->data[i] > 0.0f ? -p->data[i] / q->data[i] : 0.0f;
    return grad;
}

Tensor *tensor_kl_grad_batch(Tensor *p, Tensor *q) {
    if (!tensors_comparable(p, q)) return NULL;

    Tensor *grad = tensor_create(q->shape, q->ndim);
    float_t N = (float_t)p->shape[0];
    for (size_t i = 0; i < q->size; i++)
        grad->data[i] = p->data[i] > 0.0f ? -p->data[i] / (q->data[i] * N) : 0.0f;
    return grad;
}

static float_t kl_sum(Tensor *p, Tensor *q) {
    float_t acc = 0;
    for (size_t i = 0; i < p->size; i++) {
        if (p->data[i] > 0.0f)
            acc += p->data[i] * logf(p->data[i] / q->data[i]);
    }
    return acc;
}

float_t tensor_kl_loss(Tensor *p, Tensor *q) {
    if (!tensors_comparable(p, q)) return FLT_MAX;
    return kl_sum(p, q);
}

float_t tensor_kl_loss_batch(Tensor *p, Tensor *q) {
    if (!tensors_comparable(p, q)) return FLT_MAX;

    size_t N = p->shape[0];
    size_t C = p->size / N;
    float_t total = 0;
    for (size_t i = 0; i < N; i++) {
        float_t sample = 0;
        for (size_t j = 0; j < C; j++) {
            float_t pij = p->data[i * C + j];
            if (pij > 0.0f)
                sample += pij * logf(pij / q->data[i * C + j]);
        }
        total += sample;
    }
    return total / N;
}
