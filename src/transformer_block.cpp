//
// Created by liyif on 2025/11/29.
//
#include "transformer_block.h"

#include "weight_loader.h"
TransformerBlock::TransformerBlock(int hidden_dim, int num_heads, int ffn_dim)
    : ln1(hidden_dim),
      attn(hidden_dim, num_heads),
      ln2(hidden_dim),
      ffn(hidden_dim, ffn_dim)
{}

Tensor TransformerBlock::forward(const Tensor& x) const{
    int hidden_dim=x.rows();
    int seq_len=x.cols();

    Tensor x_norm=ln1.forward(x);

    Tensor attn_out = attn.forward(x_norm);

    Tensor x1(hidden_dim,seq_len);

    for(int i=0;i<hidden_dim;i++)
        for(int j=0;j<seq_len;j++)
            x1(i,j) = x(i,j)+attn_out(i,j);
    Tensor x_norm2=ln2.forward(x1);
    Tensor ffn_out=ffn.forward(x_norm2);
    Tensor y(hidden_dim, seq_len);
    for (int i = 0; i < hidden_dim; ++i)
        for (int t = 0; t < seq_len; ++t)
            y(i,t) = x1(i,t) + ffn_out(i,t);
    return y;
}
void TransformerBlock::load_from(WeightLoader& loader){
    // Follow the export order: attention projections -> FFN -> LayerNorms
    attn.load_from(loader);
    ffn.load_from(loader);
    ln1.load_from(loader);
    ln2.load_from(loader);
}
void TransformerBlock::enable_int8() {
    attn.enable_int8();
    ffn.enable_int8();
}