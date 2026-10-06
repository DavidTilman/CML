# CML — C Machine Learning Library

A minimal tensor library in C for building and training neural networks from scratch. No dependencies beyond the C standard library and `libm`.

## Features

- N-dimensional tensors with shape, stride, and contiguous data layout
- Element-wise arithmetic: add, sub, mul, div, pow, scale
- Linear algebra: matmul, transpose
- Reductions: sum, mean, max, min — global and per-axis
- Activations: ReLU, logistic, tanh, exp, log, sqrt, softmax
- Loss functions: MSE, MAE, Huber, BCE, CCE, Hinge, KL divergence
- Gradients for all loss functions
- Random weight initialization, fill/zeros/ones, copy and print helpers
- In-place variants of most operations (trailing underscore, e.g. `tensor_relu_`)

## Build

```sh
make        # optimized build → build/cml
make debug  # with AddressSanitizer + UBSan
make clean
./build/cml # run the example
```

Requires GCC and `libm` (standard on Linux/macOS).

## Example

`src/main.c` trains a linear model to learn `y = 3x + 2` using MSE loss and manual gradient descent:

```
epoch    0  loss: 93.769043
epoch  100  loss: 0.003842
...
learned w: 3.0000  (expected 3.0)
learned b: 2.0000  (expected 2.0)
```

## Usage

```c
#include "tensor.h"
#include "tensor_loss.h"

size_t shape[] = {2, 3};
Tensor *a = tensor_create(shape, 2);
tensor_seed(42);
tensor_rand_range(a, -1.0f, 1.0f);

Tensor *h = tensor_relu(a);        // returns a new tensor
tensor_scale_(h, 0.5f);            // trailing _ = in place
tensor_print(h);

tensor_free(h);
tensor_free(a);
```

Functions without a trailing underscore allocate and return a new tensor, which
the caller owns and must release with `tensor_free`.

## Structure

```
src/
  tensor.h        — Tensor struct and full API
  tensor_core.c   — lifecycle, access, shape ops
  tensor_math.c   — arithmetic, matmul, reductions, activations
  tensor_nn.c     — activations, softmax, init and utility helpers
  tensor_loss.h/c — loss functions and gradients
  activation.h/c  — scalar activation functions
  main.c          — example: linear regression
```
