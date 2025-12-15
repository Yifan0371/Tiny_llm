//
// Created by liyif on 2025/11/23.
//
#include "utils.h"
#include <cmath>
#include <fstream>
#include <string>
#include <omp.h>
#include <algorithm>
#include <cstdint>
#include "profiler.h"
Tensor softmax(const Tensor& x) {
    ScopedTimer timer("softmax");
    int n = x.rows();
    Tensor y(n,1);

    const float* x_ptr =x.fptr();
    float* y_ptr = y.fptr();

    float max_val = x_ptr[0];
    for (int i = 1; i < n; ++i) {
        if (max_val < x_ptr[i]) {
            max_val = x_ptr[i];
        }
	    }
    float sum = 0.0f;
    for (int i=0;i<n;i++) {
         float val =std::exp(x_ptr[i] - max_val);
         y_ptr[i] = val ;
         sum += val;
    }

    for (int i=0;i<n;i++) {
        y_ptr[i] /= sum;
    }
    return y;
}

Tensor gelu(const Tensor& x) {
    int n = x.rows();
    Tensor y(n,1);
    const float* x_ptr =x.fptr();
    float* y_ptr = y.fptr();

    for (int i=0;i<n;i++) {
        float v=x_ptr[i];
        const float kAlpha = 0.7978845608f;  // sqrt(2/pi)
        float t = std::tanh(kAlpha * (v + 0.044715f * v * v * v));
        y_ptr[i] = 0.5f * v * (1.0f + t);
    }
    return y;
}

void set_omp_threads(int num_threads) {
    if (num_threads > 0) {
        omp_set_num_threads(num_threads);
    }
}
int autotune_threads(int seq_len) {
    if (seq_len <= 32) return 2;
    if (seq_len <= 128) return 4;
    return 8;
}

void append_benchmark_csv(
    int seq_len,
    std::string precision,
    bool kv_cache,
    int threads,
    double time_ms) {
    std::ofstream fout("benchmark_results.csv", std::ios::app);
    if (!fout) {
        return;
    }

    // Write header if file is empty
    if (fout.tellp() == 0) {
        fout << "seq_len,precision,kv_cache,threads,time_ms\n";
    }

    fout << seq_len << ','
         << precision << ','
         << (kv_cache ? "true" : "false") << ','
         << threads << ','
         << time_ms << "\n";
}
float quantize_symmetric(const Tensor& tensor, std::vector<int8_t>& q_data) {
    const float* data = tensor.fptr();
    int size = tensor.size();
    q_data.resize(static_cast<std::size_t>(size));

    float max_val = 0.0f;
    for (int i = 0; i < size; ++i) {
        max_val = std::max(max_val, std::fabs(data[i]));
    }

    float scale = (max_val > 1e-8f) ? (max_val / 127.0f) : 1.0f;
    float inv_scale = 1.0f / scale;

    for (int i = 0; i < size; ++i) {
        float q = std::round(data[i] * inv_scale);
        q = std::max(-127.0f, std::min(127.0f, q));
        q_data[static_cast<std::size_t>(i)] = static_cast<int8_t>(q);
    }

    return scale;
}