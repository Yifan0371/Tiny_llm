# Tiny-LLM
Tiny-LLM 是一个使用从零实现的c++ Transformer 推理引擎，目标是在cpu环境下实现Transformer推理流程，并探索从系统层面上的性能优化

本项目不依赖于pythorch等推理框架，核心部件均由c++代码实现。

##  项目简介
Tiny-LLM 通过一个可控规模的 Tiny Transformer 模型，  
覆盖从基础算子、模型结构、权重加载，到多线程、量化与 KV-Cache 的完整推理链路。

- 自定义张量结构（Tensor）
- 核心算子（Matmul / Softmax / LayerNorm）
- 多头自注意力（Multi-Head Attention）
- 前馈网络（Feed-Forward Network）
- Transformer Block
- 完整模型推理逻辑
- CPU 优化（OpenMP、多线程、INT8 量化、KV-Cache、Auto-Tuning）
---

## 项目功能（Features）
Tiny-LLM 将实现以下核心模块：

### **1. 模型计算（Model Computation）**
- Tensor（float32 / int8）
- Linear、Softmax、LayerNorm
- Multi-Head Attention（Q/K/V + 矩阵运算 + softmax）
- Feed-Forward Network（FFN）
- Transformer Block
- 完整推理路径（Embedding → Blocks → Logits）

### **2. 权重加载（Weight Toolchain）**
- Python 构建 tiny Transformer
- 将 PyTorch 权重导出为自定义二进制文件
- C++ 加载权重并完成推理
- Python 与 C++ 推理结果数值对齐

### **3. 性能优化（Performance Optimization）**
- OpenMP 多线程
- INT8 量化推理
- 自定义 INT8 × FP32 内核
- KV-Cache（增量推理）
- Auto-Tuning（自动线程调度）

### **4. 系统级评測（Benchmarking）**
- 不同序列长度性能测试
- FP32 vs INT8 测量
- 单线程 vs 多线程分析
- KV-Cache 加速效果测试

