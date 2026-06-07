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
- Random weight initialization

## Build

```sh
make        # optimized build → build/cml
make debug  # with AddressSanitizer + UBSan
make clean
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

## Structure

```
src/
  tensor.h        — Tensor struct and full API
  tensor_core.c   — lifecycle, access, shape ops
  tensor_math.c   — arithmetic, matmul, reductions, activations
  tensor_nn.c     — neural-net helpers
  tensor_loss.c   — loss functions and gradients
  activation.h/c  — scalar activation functions
  main.c          — example: linear regression
```
