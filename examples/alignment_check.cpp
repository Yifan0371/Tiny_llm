//
// Created by liyif on 2025/12/5.
//
#include "model.h"
#include "weight_loader.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

struct ReferenceTensors {
    Tensor embedding;
    std::vector<Tensor> blocks;
    Tensor logits;
};

static void write_tensor(std::ofstream& fout, const std::string& name, const Tensor& t) {
    fout << name << ' ' << t.rows() << ' ' << t.cols() << '\n';
    for (int i = 0; i < t.size(); ++i) {
        fout << t.data()[i];
        if ((i + 1) % 8 == 0 || i + 1 == t.size()) {
            fout << '\n';
        } else {
            fout << ' ';
        }
    }
}

static Tensor read_tensor(std::ifstream& fin, const std::string& expected_name) {
    std::string name;
    int rows = 0;
    int cols = 0;
    if (!(fin >> name >> rows >> cols)) {
        throw std::runtime_error("Malformed reference file: missing header for " + expected_name);
    }
    if (name != expected_name) {
        throw std::runtime_error("Expected section '" + expected_name + "' but got '" + name + "'");
    }
    Tensor t(rows, cols);
    for (int i = 0; i < rows * cols; ++i) {
        if (!(fin >> t.data()[i])) {
            throw std::runtime_error("Malformed reference file: not enough values for " + expected_name);
        }
    }
    return t;
}

static ReferenceTensors load_reference(const std::string& path, int num_blocks) {
    std::ifstream fin(path);
    if (!fin) {
        throw std::runtime_error("Failed to open reference file: " + path);
    }

    ReferenceTensors ref;
    ref.embedding = read_tensor(fin, "embedding");
    ref.blocks.reserve(num_blocks);
    for (int i = 0; i < num_blocks; ++i) {
        ref.blocks.push_back(read_tensor(fin, "block" + std::to_string(i)));
    }
    ref.logits = read_tensor(fin, "logits");
    return ref;
}

static float max_abs_diff(const Tensor& a, const Tensor& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::runtime_error("Shape mismatch during diff computation");
    }
    float max_diff = 0.0f;
    for (int i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(a.data()[i] - b.data()[i]));
    }
    return max_diff;
}

int main(int argc, char** argv) {
    const bool dump_reference = (argc > 1 && std::string(argv[1]) == "--dump-reference");
    // Match the Python-side tiny transformer config
    const int vocab = 1000;
    const int hidden = 128;
    const int num_heads = 8;
    const int layers = 2;
    const int max_seq = 16;

    const std::string weight_path = "python/tiny_model.bin";
    const std::string reference_path = "python/reference_outputs.txt";

    std::vector<int> tokens = {10, 20, 30};

    // 1) Load weights
    WeightLoader loader(weight_path);
    TransformerModel model(vocab, hidden, num_heads, layers, max_seq);
    model.load_from(loader);

    // 2) Run forward with debug capture
    ForwardDebugInfo info = model.forward_debug(tokens);

    if (dump_reference) {
        std::ofstream fout(reference_path);
        if (!fout) {
            throw std::runtime_error("Failed to open reference output path: " + reference_path);
        }
        write_tensor(fout, "embedding", info.embedding_output);
        for (int i = 0; i < layers; ++i) {
            write_tensor(fout, "block" + std::to_string(i), info.block_outputs[i]);
        }
        write_tensor(fout, "logits", info.logits);
        std::cout << "Reference activations written to " << reference_path << "\n";
        return 0;
    }

    // 3) Load Python reference outputs
    ReferenceTensors ref = load_reference(reference_path, layers);
    if (static_cast<int>(ref.blocks.size()) != layers ||
        info.block_outputs.size() != ref.blocks.size()) {
        throw std::runtime_error("Mismatch between configured layers and reference block count");
    }

    // 4) Compare
    std::cout << "==== Alignment Report (C++ vs Python) ====\n";
    std::cout << "Embedding max |Δ|: " << max_abs_diff(info.embedding_output, ref.embedding) << "\n";
    for (int i = 0; i < layers; ++i) {
        std::cout << "Block " << i << " max |Δ|: "
                  << max_abs_diff(info.block_outputs[i], ref.blocks[i]) << "\n";
    }
    std::cout << "Logits max |Δ|: " << max_abs_diff(info.logits, ref.logits) << "\n";

    return 0;
}