//
// Created by liyif on 2025/11/29.
//
#pragma once
#include "tensor.h"
#include "linear.h"
#include "utils.h"

class FFN{
public:
    Linear fc1;
    Linear fc2;
    FFN(int hidden_dim, int intermediate_dim);

    Tensor forward(const Tensor &x)const;
};