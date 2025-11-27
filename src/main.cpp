#include <iostream>
#include "attention.h"

int main() {
    int hidden_dim = 4;
    int num_heads  = 2;
    int seq_len    = 3;

    Attention attn(hidden_dim, num_heads);

    // 简单初始化一下权重（比如都填 0.1），以后会用 load_from_file 载入真实权重
    attn.q_proj.weight.fill(0.1f);
    attn.k_proj.weight.fill(0.1f);
    attn.v_proj.weight.fill(0.1f);
    attn.o_proj.weight.fill(0.1f);

    attn.q_proj.bias.fill(0.0f);
    attn.k_proj.bias.fill(0.0f);
    attn.v_proj.bias.fill(0.0f);
    attn.o_proj.bias.fill(0.0f);

    // 构造一个简单输入 x: [hidden_dim, seq_len]
    Tensor x(hidden_dim, seq_len);
    float v = 1.0f;
    for (int c = 0; c < seq_len; ++c) {
        for (int r = 0; r < hidden_dim; ++r) {
            x(r, c) = v++;
        }
    }

    Tensor y = attn.forward(x);

    std::cout << "Attention output:" << std::endl;
    for (int r = 0; r < hidden_dim; ++r) {
        for (int c = 0; c < seq_len; ++c) {
            std::cout << y(r, c) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
