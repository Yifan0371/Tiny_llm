//
// Created by liyif on 2025/11/22.
//
// src/tensor.cpp
#include "tensor.h"
#include <fstream>
#include <stdexcept>  // for std::out_of_range
#include <algorithm>
#include <cmath>
Tensor::Tensor() : rows_(0), cols_(0), data_() {}

// 构造函数：一创建就分配好 rows * cols 的内存
Tensor::Tensor(int rows, int cols)
    : rows_(rows), cols_(cols), data_(rows * cols) {
    // 默认元素值是 0.0f（vector 默认初始化）
}

void Tensor::resize(int rows, int cols) {
    rows_ = rows;
    cols_ = cols;
    data_.assign(rows * cols, 0.0f);  // 重置为新大小，并全部置 0
    // 这里用 assign 而不是 resize，是为了确保旧数据清零，逻辑更简单
}

float& Tensor::operator()(int r, int c) {
    // 这里可以做一个简单的边界检查，方便 debug
#ifndef NDEBUG
    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        throw std::out_of_range("Tensor index out of range");
    }
#endif
    return data_[static_cast<std::size_t>(r) * cols_ + c];
}

const float& Tensor::operator()(int r, int c) const {
#ifndef NDEBUG
    if (r < 0 || r >= rows_ || c < 0 || c >= cols_) {
        throw std::out_of_range("Tensor index out of range");
    }
#endif
    return data_[static_cast<std::size_t>(r) * cols_ + c];
}

void Tensor::fill(float value) {
    std::fill(data_.begin(), data_.end(), value);
}

void Tensor::save_to_file(const std::string& path) const{
	std::ofstream fout(path, std::ios::binary);//写入二进制文件
	if (!fout.is_open()) {
	throw std::runtime_error("Could not open file " + path);
	}
    fout.write(reinterpret_cast<const char*>(data_.data()),data_.size() * sizeof(float));//使用reinterpret强换为const char*
}


void Tensor::load_from_file(const std::string& path){
	std::ifstream fin(path, std::ios::binary);
    if (!fin) {throw std::runtime_error("Failed to open file for reading: " + path);}
    fin.read(reinterpret_cast<char*>(data_.data()),data_.size() * sizeof(float));
}
QuantizedTensor quantize_tensor(const Tensor& src) {
    QuantizedTensor q(src.rows(), src.cols());

    const float* ptr = src.fptr();
    float max_abs = 0.0f;
    for (std::size_t i = 0; i < src.size(); ++i) {
        max_abs = std::max(max_abs, std::fabs(ptr[i]));
    }

    q.scale = (max_abs < 1e-8f) ? 1e-8f : max_abs / 127.0f;

    for (std::size_t i = 0; i < src.size(); ++i) {
        float scaled = ptr[i] / q.scale;
        scaled = std::max(-127.0f, std::min(127.0f, std::round(scaled)));
        q.data[i] = static_cast<int8_t>(scaled);
    }

    return q;
}

Tensor dequantize_tensor(const QuantizedTensor& q) {
    Tensor out(q.rows, q.cols);
    float* dst = out.fptr();
    for (std::size_t i = 0; i < out.size(); ++i) {
        dst[i] = static_cast<float>(q.data[i]) * q.scale;
    }
    return out;
}