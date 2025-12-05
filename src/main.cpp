#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include "transformer_block.h"
#include "weight_loader.h"

// For Day 13: demonstrate loading a full set of weights exported by Python
// into our C++ inference stack.

static void write_dummy_weights(const std::string& path,
                                int hidden_dim,
                                int num_heads,
                                int ffn_dim,
                                int vocab_size) {
    std::ofstream fout(path, std::ios::binary);
    if (!fout) {
        throw std::runtime_error("Failed to create dummy weight file");
    }

    float value = 0.0f;
    auto write_block = [&](std::size_t count) {
        std::vector<float> buf(count);
        for (std::size_t i = 0; i < count; ++i) {
            buf[i] = value++;
        }
        fout.write(reinterpret_cast<const char*>(buf.data()),
                   static_cast<std::streamsize>(count * sizeof(float)));
    };

    // Embedding table
    write_block(static_cast<std::size_t>(hidden_dim) * vocab_size);

    // Attention projections: q, k, v, out (each weight then bias)
    for (int i = 0; i < 4; ++i) {
        write_block(static_cast<std::size_t>(hidden_dim) * hidden_dim); // weight
        write_block(hidden_dim);                                        // bias
    }

    // FFN layers: fc1 then fc2
    write_block(static_cast<std::size_t>(ffn_dim) * hidden_dim); // fc1 weight
    write_block(ffn_dim);                                        // fc1 bias
    write_block(static_cast<std::size_t>(hidden_dim) * ffn_dim); // fc2 weight
    write_block(hidden_dim);                                     // fc2 bias

    // LayerNorms: ln1 then ln2 (gamma then beta)
    write_block(hidden_dim); // ln1 gamma
    write_block(hidden_dim); // ln1 beta
    write_block(hidden_dim); // ln2 gamma
    write_block(hidden_dim); // ln2 beta
}
int main() {
    int hidden_dim = 4;
    int num_heads  = 2;

    int ffn_dim    = 8;
    int seq_len    = 3;
    int vocab_size = 6;

    const std::string weight_path = "dummy_weights.bin";
    write_dummy_weights(weight_path, hidden_dim, num_heads, ffn_dim, vocab_size);

    // Allocate model parameters
    Tensor embedding(hidden_dim, vocab_size);
    TransformerBlock block(hidden_dim, num_heads, ffn_dim);

    // Load the parameters in the exact order the Python exporter would write them
    WeightLoader loader(weight_path);
    loader.read_into(embedding);
    block.load_from(loader);

    // Build a toy input and run the forward pass
    Tensor x(hidden_dim, seq_len);
    float v = 1.0f;
    for (int c = 0; c < seq_len; ++c) {
        for (int r = 0; r < hidden_dim; ++r) {
            x(r, c) = v++;
        }
    }

    Tensor y = block.forward(x);


    std::cout << "Loaded embedding sample: " << embedding(0, 0)
              << ", " << embedding(0, 1) << "\n";

    std::cout << "TransformerBlock Output with loaded weights:\n";
    for (int r = 0; r < hidden_dim; ++r) {
        for (int c = 0; c < seq_len; ++c) {
            std::cout << y(r, c) << " ";
        }
        std::cout << "\n";
    }

    return 0;
}