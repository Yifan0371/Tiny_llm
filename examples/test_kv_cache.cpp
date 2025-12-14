//
// Created by liyif on 2025/12/14.
//
#include "model.h"
#include "utils.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main() {
    // Make initialization deterministic so full vs. incremental are comparable.
    std::srand(42);
    set_omp_threads(2);

    const int vocab = 32;
    const int hidden = 16;
    const int heads = 4;
    const int layers = 2;
    const int max_seq_len = 8;

    TransformerModel model(vocab, hidden, heads, layers, max_seq_len);

    // Short token sequence for comparison.
    std::vector<int> tokens = {1, 3, 5, 7};

    Tensor full_logits = model.forward(tokens);

    std::vector<KVCache> caches;
    Tensor incremental_logits = model.forward_incremental(tokens, caches);

    if (full_logits.rows() != incremental_logits.rows() ||
        full_logits.cols() != incremental_logits.cols()) {
        std::cerr << "Shape mismatch between full and incremental logits" << std::endl;
        return 1;
        }

    float max_diff = 0.0f;
    for (int r = 0; r < full_logits.rows(); ++r) {
        for (int c = 0; c < full_logits.cols(); ++c) {
            float diff = std::fabs(full_logits(r, c) - incremental_logits(r, c));
            if (diff > max_diff) {
                max_diff = diff;
            }
        }
    }

    std::cout << "Max difference between full and incremental logits: "
              << max_diff << std::endl;

    const float kTolerance = 1e-4f;
    if (max_diff > kTolerance) {
        std::cerr << "KV cache incremental path diverges from full forward" << std::endl;
        return 1;
    }

    std::cout << "KV cache incremental attention test passed" << std::endl;
    return 0;
}