#include <iostream>
#include "transformer_block.h"

int main() {
    int hidden_dim = 4;
    int num_heads  = 2;
    int ffn_dim    = 16;
    int seq_len    = 3;

    TransformerBlock block(hidden_dim, num_heads, ffn_dim);

    Tensor x(hidden_dim, seq_len);
    float v = 1.0f;

    for (int c = 0; c < seq_len; ++c) {
        for (int r = 0; r < hidden_dim; ++r) {
            x(r, c) = v++;
        }
    }

    Tensor y = block.forward(x);

    std::cout << "TransformerBlock Output:\n";
    for (int r = 0; r < hidden_dim; ++r) {
        for (int c = 0; c < seq_len; ++c) {
            std::cout << y(r, c) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}
