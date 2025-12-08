//
// Created by liyif on 2025/12/14.
//

#pragma once

#include "tensor.h"
#include "linear.h"
#include <vector>
#include <cstdint>

class LinearInt8 {
public:
    explicit LinearInt8(const Linear& linear_fp32);

    Tensor forward(const Tensor& x) const;

public:
    int out_dim;
    int in_dim;

    std::vector<int8_t> weight_i8;  // [out_dim * in_dim]
    float scale;                     // weight scale
    Tensor bias;                     // bias kept in FP32
};