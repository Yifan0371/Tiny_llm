//
// Created by liyif on 2025/11/29.
//
#pragma once
#include "tensor.h"
#include "linear.h"
#include "utils.h"

class WeightLoader;


class FFN{
public:
    Linear fc1;
    Linear fc2;
    FFN(int hidden_dim, int intermediate_dim);

    Tensor forward(const Tensor &x)const;

    // Load FC1 and FC2 parameters from the shared weight stream.
    void load_from(WeightLoader& loader);
};