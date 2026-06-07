#include "tensor.h"
#include "tensor_loss.h"

#include <stdio.h>
#include <string.h>

int main() {
    // learn y = 3x + 2

    size_t N = 5                           ; // num points
    float x_data[] = {  1,  2,  3,  4,  5 }; // x points
    float y_data[] = {  5,  8, 11, 14, 17 }; // y values

    size_t x_shape[] = {N, 1}; // [n points]
    size_t w_shape[] = {1, 1}; // weight
    size_t b_shape[] = {   1}; // bias
    size_t y_shape[] = {N, 1}; // [n points]

    Tensor *X = tensor_create(x_shape, 2);
    Tensor *w = tensor_create(w_shape, 2);
    Tensor *b = tensor_create(b_shape, 1);
    Tensor *y = tensor_create(y_shape, 2);

    // fill data
    memcpy(X->data, x_data, X->size * sizeof(float));
    memcpy(y->data, y_data, y->size * sizeof(float));

    tensor_seed(42);

    // start with random weights
    tensor_rand_range(w, -0.1f, 0.1f);
    tensor_fill(b, 0.0f);


    // learning rate
    float lr = 0.01f;

    for (int e = 0; e < 1000; e++) {
        // forward: y_pred = X @ w + b
        Tensor *xw     = tensor_matmul(X, w);
        Tensor *y_pred = tensor_add(xw, b);

        float loss = tensor_mse_loss(y_pred, y);

        if (e % 100 == 0)
            printf("epoch %4d  loss: %.6f\n", e, loss);

        // backward
        Tensor *dl    = tensor_mse_grad(y_pred, y);   // [N, 1]
        Tensor *xt    = tensor_transpose(X);          // [1, N]
        Tensor *dl_dw = tensor_matmul(xt, dl);        // [1, 1]
        Tensor *dl_db = tensor_sum_axis(dl, 0);       // [1]

        for (size_t i = 0; i < w->size; i++) w->data[i] -= lr * dl_dw->data[i];
        for (size_t i = 0; i < b->size; i++) b->data[i] -= lr * dl_db->data[i];

        tensor_free(xw);
        tensor_free(y_pred);
        tensor_free(dl);
        tensor_free(xt);
        tensor_free(dl_dw);
        tensor_free(dl_db);
    }

    printf("\nlearned w: %.4f  (expected 3.0)\n", w->data[0]);
    printf("learned b: %.4f  (expected 2.0)\n", b->data[0]);

    tensor_free(X);
    tensor_free(w);
    tensor_free(b);
    tensor_free(y);

    return 0;
}
