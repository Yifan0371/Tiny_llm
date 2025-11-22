# Tiny-LLM
### A Lightweight CPU-based Transformer Inference Engine (C++ from Scratch)

##  项目简介（What is Tiny-LLM?）
Tiny-LLM 是一个 **基于 C++ 从零实现的轻量级 Transformer 推理引擎**，专门针对 **CPU 环境** 进行优化。
项目的核心目标是完整复现 Transformer 模型的推理流程，并构建一个具备 **高性能、可扩展、可分析** 的推理系统。

Tiny-LLM **不依赖 PyTorch、TensorRT、ONNX Runtime** 等框架，所有实现均为手写，包括：

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
- 输出 CSV 用于绘图和报告

