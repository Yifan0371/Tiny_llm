//
// Created by liyif on 2025/12/15.
//
#include "model.h"
#include "profiler.h"
#include "utils.h"

#include <chrono>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
                  << " <weight_file> <vocab_size> <hidden_dim> <num_heads> <num_layers> [max_seq_len] [iterations]\n";
        return 1;
    }

    std::string weight_path = argv[1];
    int vocab_size = std::stoi(argv[2]);
    int hidden_dim = std::stoi(argv[3]);
    int num_heads = std::stoi(argv[4]);
    int num_layers = std::stoi(argv[5]);
    int max_seq_len = (argc > 6) ? std::stoi(argv[6]) : 256;
    int iterations = (argc > 7) ? std::stoi(argv[7]) : 10;

    TransformerModel model(vocab_size, hidden_dim, num_heads, num_layers, max_seq_len);

    WeightLoader loader(weight_path);
    model.load_from(loader);

    const std::vector<int> seq_lens = {8, 32, 128, 256};

    for (int seq_len : seq_lens) {
        if (seq_len > max_seq_len) {
            std::cerr << "Skipping seq_len=" << seq_len << " because it exceeds max_seq_len." << std::endl;
            continue;
        }

        std::vector<int> tokens;
        tokens.reserve(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            tokens.push_back(i % vocab_size);
        }

        int threads = autotune_threads(seq_len);
        omp_set_num_threads(threads);

        auto start = std::chrono::steady_clock::now();
        for (int iter = 0; iter < iterations; ++iter) {
            model.forward(tokens);
        }
        auto end = std::chrono::steady_clock::now();

        double total_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        double avg_ms = total_ms / iterations;

        append_benchmark_csv(seq_len, "fp32", false, threads, avg_ms);
        std::cout << "seq_len=" << seq_len << ", threads=" << threads
                  << ", total_ms=" << total_ms << ", avg_ms=" << avg_ms << std::endl;
    }

    dump_profile_csv("profile.csv");
    return 0;
}