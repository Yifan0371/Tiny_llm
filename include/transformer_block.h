//
// Created by liyif on 2025/11/29.
//
#pragma once
#include "layernorm.h"
#include "attention.h"
#include "ffn.h"
#include "tensor.h"

class TransformerBlock{
public:
    LayerNorm ln1;
    MultiHeadAttention mha;
    LayerNorm ln2;
    FFN ffn;
    TransformerBlock(int hidden_dim, int intermediate_dim);
    Tensor forward(const Tensor& x) const;
};