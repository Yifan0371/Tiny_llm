//
// Created by liyif on 2025/11/24.
//
#pragma once

#include "tensor.h"
#include "linear.h"

class WeightLoader;
struct KVCache {
    Tensor k_cache;
    Tensor v_cache;
};

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
    Tensor forward_incremental(const Tensor& x, Tensor& k_cache, Tensor& v_cache) const;
    void enable_int8();
	    // Load q/k/v/o projection weights from a shared binary stream.
    void load_from(WeightLoader& loader);
};