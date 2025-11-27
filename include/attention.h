//
// Created by liyif on 2025/11/24.
//
#pragma once

#include "Tensor.h"
#include "linear.h"


class Attention{
public:
    int num_heads;
    int head_dim;

    Linear q_proj;
    Linear k_proj;
    Linear v_proj;
    Linear o_proj;

    Attention(int hidden_dim, int num_heads);
    Tensor forward(const Tensor & x) const;
};