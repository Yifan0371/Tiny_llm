#pragma once
#include "tensor.h"
#include "layernorm.h"
#include "ffn.h"
#include "attention.h"

class TransformerBlock {
public:
    LayerNorm ln1;
    Attention attn;
    LayerNorm ln2;
    FFN ffn;

    TransformerBlock(int hidden_dim, int num_heads, int ffn_dim);
    Tensor forward(const Tensor& x) const;
};
