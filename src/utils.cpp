//
// Created by liyif on 2025/11/23.
//
#include "utils.h"
#include <cmath>

#include <omp.h>
#include <algorithm>
#include <cstdint>
Tensor softmax(const Tensor& x) {
    int n = x.rows();
    Tensor y(n,1);

    const float* x_ptr =x.fptr();
    float* y_ptr = y.fptr();

    float max_val = x_ptr[0];
    for (int i = 1; i < n; ++i) {
        if (max_val < x_ptr[i]) {
            max_val = x_ptr[i];
        }
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

Tensor gelu(const Tensor& x) {
    int n = x.rows();
    Tensor y(n,1);
    const float* x_ptr =x.fptr();
    float* y_ptr = y.fptr();

    for (int i=0;i<n;i++) {
        float v=x_ptr[i];
        const float kAlpha = 0.7978845608f;  // sqrt(2/pi)
        float t = std::tanh(kAlpha * (v + 0.044715f * v * v * v));
        y_ptr[i] = 0.5f * v * (1.0f + t);
    }
    return y;
}

void set_omp_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}

float quantize_symmetric(const Tensor& tensor, std::vector<int8_t>& q_data) {
    const float* data = tensor.fptr();
    int size = tensor.size();
    q_data.resize(static_cast<std::size_t>(size));

    float max_val = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_val = std::max(max_val, std::fabs(data[i]));
    }

    float scale = (max_val > 1e-8f) ? (max_val / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < size; ++i) {
        float q = std::round(data[i] * inv_scale);
        q = std::max(-127.0f, std::min(127.0f, q));
        q_data[static_cast<std::size_t>(i)] = static_cast<int8_t>(q);
    }

    return scale;
}