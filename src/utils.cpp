//
// Created by liyif on 2025/11/23.
//
#include "utils.h"
#include <cmath>
#include <algorithm>

Tensor softmax(const Tensor& x) {
    int n = x.rows();
    Tensor y(n,1);

    const float* x_ptr =x.fptr();
    float* y_ptr = y.fptr();

    float max_val=x_ptr[0];
    for (int i=0;i<n-1;i++) {
        if (max_val<x_ptr[i]) {max_val=x_ptr[i];}
    }
    float sum = 0.0f;
    for (int i=0;i<n;i++) {
         float val =std::exp(x_ptr[i] - max_val);
         y_ptr[i] = val ;
         sum += val;
    }

    for (int i=0;i<n;i++) {
        y_ptr[i] /= sum;
    }
    return y;
}
