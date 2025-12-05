//
// Created by liyif on 2025/12/5.
//
#pragma once

#include <fstream>
#include <string>
#include "tensor.h"

// A tiny helper to read float32 weights from a single binary stream
// in row-major order into pre-sized Tensor objects.
class WeightLoader {
public:
    explicit WeightLoader(const std::string& path);

    // Read exactly tensor.size() float values from the stream
    // and write them into the tensor's underlying storage.
    void read_into(Tensor& tensor);

private:
    std::ifstream fin_;
};