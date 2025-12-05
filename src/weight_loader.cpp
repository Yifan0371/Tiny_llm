//
// Created by liyif on 2025/12/5.
//
#include "weight_loader.h"

#include <stdexcept>
#include <sstream>

WeightLoader::WeightLoader(const std::string& path) : fin_(path, std::ios::binary) {
    if (!fin_) {
        throw std::runtime_error("Failed to open weight file: " + path);
    }
}

void WeightLoader::read_into(Tensor& tensor) {
    const std::size_t bytes = tensor.size() * sizeof(float);
    fin_.read(reinterpret_cast<char*>(tensor.data()), static_cast<std::streamsize>(bytes));
    if (!fin_) {
        std::ostringstream oss;
        oss << "Failed to read " << tensor.size() << " floats from weight file.";
        throw std::runtime_error(oss.str());
    }
}