//
// Created by liyif on 2025/11/22.
//
#pragma once
#include "tensor.h"
class WeightLoader;

class Linear{
public:
    Tensor weight;// [out_dim, in_dim]
    Tensor bias;  // [out_dim]

    Linear(int out_dim,int in_dim);
    Tensor forward(const Tensor& x) const;

    // Load weight and bias values from a shared binary weight stream.
    void load_from(WeightLoader& loader);
};
