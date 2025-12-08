//
// Created by liyif on 2025/11/23.
//
#pragma once
#include "tensor.h"

Tensor softmax(const Tensor& x);
Tensor gelu(const Tensor& x);

// Configure OpenMP threading for the library. Exposed to keep examples self-contained
// for Day 16 multi-threading experiments.
void set_omp_threads(int num_threads);