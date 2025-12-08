//
// Created by liyif on 2025/12/5.
//
#include "model.h"
#include "weight_loader.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <iomanip>

struct ReferenceTensors {
    Tensor embedding;
    std::vector<Tensor> blocks;
    Tensor logits;
};

struct DiffResult {
    float max_abs_diff = 0.0f;
    int row = -1;
    int col = -1;
    float a_val = 0.0f;
    float b_val = 0.0f;
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

static DiffResult compute_diff(const Tensor& a, const Tensor& b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        throw std::runtime_error("Shape mismatch during diff computation");
    }
    DiffResult res;
    res.max_abs_diff = 0.0f;
    const int rows = a.rows();
    const int cols = a.cols();

    for (int i = 0; i < a.size(); ++i) {
        float da = a.data()[i];
        float db = b.data()[i];
        float diff = std::fabs(da - db);
        if (diff > res.max_abs_diff) {
            res.max_abs_diff = diff;
            res.row = i / cols;
            res.col = i % cols;
            res.a_val = da;
            res.b_val = db;
        }
    }
    return res;
}

static void print_diff_report(const std::string& name,
                              const DiffResult& diff,
                              float tolerance) {
    std::cout << std::fixed << std::setprecision(8);
    std::cout << name << " max |Δ|: " << diff.max_abs_diff;
    if (diff.row >= 0 && diff.col >= 0) {
        std::cout << " at (" << diff.row << ", " << diff.col << ") "
                  << "[C++=" << diff.a_val << ", Py=" << diff.b_val << "]";
    }
    if (diff.max_abs_diff > tolerance) {
        std::cout << "  [WARN: exceeds tol=" << tolerance << "]";
    }
    std::cout << "\n";
}

int main(int argc, char** argv) {

    set_omp_threads(4);

    // 控制是否由 C++ 侧导出“参考输出”（一般还是建议由 Python 写参考）
    const bool dump_reference = (argc > 1 && std::string(argv[1]) == "--dump-reference");

    // Match the Python-side tiny transformer config
    const int vocab = 1000;
    const int hidden = 128;
    const int num_heads = 8;
    const int layers = 2;
    const int max_seq = 16;

    const std::string weight_path = "python/tiny_model.bin";
    const std::string reference_path = "python/reference_outputs.txt";

    // 和 Python 对齐时要保证 tokens 序列一致
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
    constexpr float TOLERANCE = 1e-4f;

    std::cout << "==== Alignment Report (C++ vs Python) ====\n";
    std::cout << "Tokens: ";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << (i + 1 == tokens.size() ? "" : ", ");
    }
    std::cout << "\n";

    bool all_ok = true;

    DiffResult emb_diff = compute_diff(info.embedding_output, ref.embedding);
    print_diff_report("Embedding", emb_diff, TOLERANCE);
    all_ok = all_ok && (emb_diff.max_abs_diff <= TOLERANCE);

    for (int i = 0; i < layers; ++i) {
        DiffResult blk_diff = compute_diff(info.block_outputs[i], ref.blocks[i]);
        print_diff_report("Block " + std::to_string(i), blk_diff, TOLERANCE);
        all_ok = all_ok && (blk_diff.max_abs_diff <= TOLERANCE);
    }

    DiffResult logit_diff = compute_diff(info.logits, ref.logits);
    print_diff_report("Logits", logit_diff, TOLERANCE);
    all_ok = all_ok && (logit_diff.max_abs_diff <= TOLERANCE);

    std::cout << "-----------------------------------------\n";
    if (all_ok) {
        std::cout << "Alignment status: PASS (all max |Δ| ≤ " << TOLERANCE << ")\n";
        return 0;
    } else {
        std::cout << "Alignment status: FAIL (some max |Δ| > " << TOLERANCE << ")\n";
        return 1;
    }
}
