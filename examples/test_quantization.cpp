//
// Created by liyif on 2025/12/14.
//

#include <iostream>
#include <cmath>
#include "linear.h"
#include "linear_int8.h"
#include "utils.h"

int main() {
    int in_dim = 256;
    int out_dim = 256;

    Linear lin_fp32(out_dim, in_dim);

    std::cout << "Quantizing weights..." << std::endl;
    LinearInt8 lin_int8(lin_fp32);
    std::cout << "Scale factor: " << lin_int8.scale << std::endl;

    Tensor x(in_dim, 1);
    for (int i = 0; i < in_dim; ++i) {
        x(i, 0) = static_cast<float>(rand()) / RAND_MAX;
    }

    Tensor y_fp32 = lin_fp32.forward(x);
    Tensor y_int8 = lin_int8.forward(x);

    float max_diff = 0.0f;
    float total_diff = 0.0f;
    for (int i = 0; i < out_dim; ++i) {
        float diff = std::fabs(y_fp32(i, 0) - y_int8(i, 0));
        max_diff = std::max(max_diff, diff);
        total_diff += diff;
    }

    std::cout << "Max Absolute Error: " << max_diff << std::endl;
    std::cout << "Avg Absolute Error: " << total_diff / out_dim << std::endl;

    if (max_diff < 0.1f) {
        std::cout << "SUCCESS: Quantization error is within acceptable range." << std::endl;
    } else {
        std::cout << "WARNING: Large quantization error detected." << std::endl;
    }

    return 0;
}