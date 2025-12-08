//
// Created by liyif on 2025/12/14.
//
#include "linear_int8.h"
#include "utils.h"
#include <omp.h>
#include <cassert>

LinearInt8::LinearInt8(const Linear& linear_fp32)
    : bias(linear_fp32.bias) {
    out_dim = linear_fp32.weight.rows();
    in_dim = linear_fp32.weight.cols();

    scale = quantize_symmetric(linear_fp32.weight, weight_i8);
}

Tensor LinearInt8::forward(const Tensor& x) const {
    assert(x.cols() == 1);
    assert(x.rows() == in_dim);

    Tensor y(out_dim, 1);
    const float* x_ptr = x.fptr();
    const int8_t* w_ptr = weight_i8.data();
    const float* b_ptr = bias.fptr();
    float* y_ptr = y.fptr();

    // y = scale * (W_int8 * x) + b
#pragma omp parallel for
    for (int i = 0; i < out_dim; ++i) {
        float acc = 0.0f;
        int offset = i * in_dim;
        for (int j = 0; j < in_dim; ++j) {
            acc += static_cast<float>(w_ptr[offset + j]) * x_ptr[j];
        }
        y_ptr[i] = acc * scale + b_ptr[i];
    }

    return y;
}
