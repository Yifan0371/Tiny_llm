//
// Created by liyif on 2025/11/22.
//
#include "linear.h"
#include <cassert>
#include <cstdlib>
#include <cmath>
//Xavier初始化
Linear::Linear(int out_dim,int in_dim)
    : weight(out_dim,in_dim), bias(out_dim,1)
{
    float scale = std::sqrt(6.0f / (out_dim + in_dim));
    for (int i = 0; i < out_dim; i++) {
        for (int j = 0; j < in_dim; j++) {
            float r = (float)rand() / RAND_MAX;  // [0,1]
            weight(i,j) = (r * 2 - 1) * scale;   // [-scale, scale]
        }
        bias(i,0) = 0.0f;
    }
}
Tensor Linear::forward(const Tensor& x) const {
    assert(x.cols() == 1);   // x 是列向量
    assert(weight.cols() == x.rows());

    int out_dim = weight.rows();
    int in_dim  = weight.cols();

    Tensor y(out_dim, 1);

    const float* w_ptr = weight.fptr();
    const float* x_ptr = x.fptr();
    const float* b_ptr = bias.fptr();
    float* y_ptr = y.fptr();

    for (int i = 0; i < out_dim; i++) {
        float sum = b_ptr[i];
        for (int j = 0; j < in_dim; j++) {
            sum += w_ptr[i * in_dim + j] * x_ptr[j];
        }
        y_ptr[i] = sum;
    }

    return y;
}
