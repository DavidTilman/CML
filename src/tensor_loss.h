#ifndef TENSOR_LOSS_H
#define TENSOR_LOSS_H

#include <math.h>
#include "tensor.h"

float_t  tensor_mse_loss(Tensor *a, Tensor *b);
float_t  tensor_mae_loss(Tensor *a, Tensor *b);
float_t  tensor_huber_loss(Tensor *a, Tensor *b, float_t delta);
float_t  tensor_bce_loss(Tensor *a, Tensor *b);
float_t  tensor_cce_loss(Tensor *a, Tensor *b);
float_t  tensor_hinge_loss(Tensor *a, Tensor *b);
float_t  tensor_kl_loss(Tensor *p, Tensor *q);
float_t  tensor_kl_loss_batch(Tensor *p, Tensor *q);

Tensor  *tensor_mse_grad(Tensor *a, Tensor *b);
Tensor  *tensor_mae_grad(Tensor *a, Tensor *b);
Tensor  *tensor_huber_grad(Tensor *a, Tensor *b, float_t delta);
Tensor  *tensor_bce_grad(Tensor *a, Tensor *b);
Tensor  *tensor_cce_grad(Tensor *a, Tensor *b);
Tensor  *tensor_hinge_grad(Tensor *a, Tensor *b);
Tensor  *tensor_kl_grad(Tensor *p, Tensor *q);
Tensor  *tensor_kl_grad_batch(Tensor *p, Tensor *q);

#endif
