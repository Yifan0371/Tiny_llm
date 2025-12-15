//
// Created by liyif on 2025/11/23.
//
#pragma once
#include "tensor.h"
#include <cstdint>
#include <string>
#include <vector>
Tensor softmax(const Tensor& x);
Tensor gelu(const Tensor& x);

// Configure OpenMP threading for the library. Exposed to keep examples self-contained
// for Day 16 multi-threading experiments.
void set_omp_threads(int num_threads);
int autotune_threads(int seq_len);

// Append a benchmark record to benchmark_results.csv.
void append_benchmark_csv(
    int seq_len,
    std::string precision,
    bool kv_cache,
    int threads,
    double time_ms);

// Symmetric per-tensor quantization helper. Returns the scale used to quantize the
// values in `tensor` into the provided `q_data` buffer (range [-127, 127]).
float quantize_symmetric(const Tensor& tensor, std::vector<int8_t>& q_data);