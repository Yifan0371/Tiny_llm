//
// Created by liyif on 2025/11/22.
//
#pragma once
#include <vector>
#include <cstdint>
#include "tensor.h"
class WeightLoader;

class Linear{
public:
    Tensor weight;// [out_dim, in_dim]
    Tensor bias;  // [out_dim]

    Linear(int out_dim,int in_dim);
    Tensor forward(const Tensor& x) const;
    void enable_int8();
    bool is_int8_enabled() const { return use_int8_; }
    // Load weight and bias values from a shared binary weight stream.
    void load_from(WeightLoader& loader);

private:
    Tensor forward_int8(const Tensor& x) const;
    void quantize_weight();

    std::vector<int8_t> weight_int8_;
    float weight_scale_{1.0f};
    bool use_int8_{false};
};
