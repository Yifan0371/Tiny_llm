//
// Created by liyif on 2025/11/22.
//include/tensor.h
//

#pragma once
#include <string>
#include <vector>
#include <cstddef>

// 一个简单的 2D float32 Tensor：
// - 行主序存储 (row-major)
// - 数据存在 std::vector<float> 中
// - 通过 (row, col) 访问元素
class Tensor {
public:
    // 默认构造：得到一个 0x0 的空 Tensor
    Tensor();

    // 指定行列数的构造函数，会分配 rows * cols 个 float
    Tensor(int rows, int cols);

    // 改变 Tensor 的形状，会重新分配空间（丢弃旧内容）
    void resize(int rows, int cols);

    // 获取行数、列数
    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // 返回底层数据指针（可读写）
    float* data() { return data_.data(); }

    // 返回底层数据指针（只读版本）
    const float* data() const { return data_.data(); }

    // 与你计划中的 fptr() 对应：返回 float* 指针
    float* fptr() { return data(); }
    const float* fptr() const { return data(); }

    // 方便访问元素：tensor(r, c)
    float& operator()(int r, int c);
    const float& operator()(int r, int c) const;

    // 把所有元素填成同一个值（例如 0.0f）
    void fill(float value);

    // 当前元素总个数（rows * cols）
    std::size_t size() const { return data_.size(); }
	//从文件读float32数据
	void load_from_file(const std::string& path);
	//写入float32二进制文件
	void save_to_file(const std::string& path) const;

private:
    int rows_{0};
    int cols_{0};
    std::vector<float> data_;  // 行主序存储的连续内存
};
