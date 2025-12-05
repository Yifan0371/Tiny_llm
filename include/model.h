//
// Created by liyif on 2025/12/1.
//
#pragma once
#include <vector>
#include "tensor.h"
#include "linear.h"
#include "attention.h"
#include "ffn.h"
#include "layernorm.h"
#include "transformer_block.h"
#include "weight_loader.h"

struct ForwardDebugInfo {
    Tensor embedding_output;
    std::vector<Tensor> block_outputs;
    Tensor logits;
};
class TransformerModel{
public:
    //超参数
    int vocab_size;//词表大小
    int hidden_dim;//隐藏维度
    int num_heads;//注意力多头
    int num_layers;//transformer 层数
    int ffn_dim;//ffn维度
    int max_seq_len;//最大支持的序列长度

    //模型参数
    Tensor embedding;
    //N层的transformer
    std::vector<TransformerBlock> blocks;
    //输出层
    Linear lm_head;
    //ffn默认输出维度是4
    TransformerModel(int vocab_size,
                     int hidden_dim,
                     int num_heads,
                     int num_layers,
                     int max_seq_len);
    // ====== 前向推理 ======
    // 输入: token 序列（长度 T）
    // 输出: logits，形状 [vocab_size, T]
    Tensor forward(const std::vector<int>& tokens) const;
	void load_from(WeightLoader& loader);
	 ForwardDebugInfo forward_debug(const std::vector<int>& tokens) const;
};