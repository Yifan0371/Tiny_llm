//
// Created by liyif on 2025/12/1.
//
#include "model.h"
#include <cassert>
#include <omp.h>
#include "utils.h"

// 构造函数实现
TransformerModel::TransformerModel(int vocab_size_,
                                   int hidden_dim_,
                                   int num_heads_,
                                   int num_layers_,
                                   int max_seq_len_)
    : vocab_size(vocab_size_),
      hidden_dim(hidden_dim_),
      num_heads(num_heads_),
      num_layers(num_layers_),
      ffn_dim(hidden_dim_ * 4),   // 这里简单地设为 4 * hidden_dim，你可以改
      max_seq_len(max_seq_len_),
      // embedding: [vocab_size, hidden_dim]
      embedding(vocab_size_, hidden_dim_),
      // lm_head: out_dim = vocab_size, in_dim = hidden_dim
      lm_head(vocab_size_, hidden_dim_)
{
    blocks.reserve(num_layers);
    for (int i = 0; i < num_layers; i++){
        blocks.emplace_back(hidden_dim_,num_heads_,ffn_dim);
    }
}

void TransformerModel::load_from(WeightLoader& loader) {
    loader.read_into(embedding);
    for (auto& block : blocks) {
        block.load_from(loader);
    }
    lm_head.load_from(loader);
}


void TransformerModel::enable_int8() {
    for (auto& block : blocks) {
        block.enable_int8();
    }
    lm_head.enable_int8();
}
Tensor TransformerModel::forward(const std::vector<int>& tokens) const
{
    return forward_debug(tokens).logits;
}

ForwardDebugInfo TransformerModel::forward_debug(const std::vector<int>& tokens) const {
    int T = static_cast<int>(tokens.size());
    assert(T > 0 && T <= max_seq_len);
    int threads = autotune_threads(T);
    omp_set_num_threads(threads);
    ForwardDebugInfo info;
    // ===========================
    // 1. Embedding lookup: [hidden_dim, T]
    // ===========================
    Tensor x(hidden_dim, T);

    for (int t = 0; t < T; ++t) {
        int id = tokens[t];
        assert(id >= 0 && id < vocab_size);

        for (int h = 0; h < hidden_dim; ++h) {
            // embedding[id, h]
            x(h, t) = embedding(id, h);
        }
    }
	info.embedding_output = x;
    // ===========================
    // 2. Pass through N TransformerBlocks
    // ===========================
    Tensor h = x;
    for (const auto& block : blocks) {
        h = block.forward(h);
		info.block_outputs.push_back(h);  // shape still [hidden_dim, T]
    }

    // ===========================
    // 3. lm_head: per-token linear projection
    //    Input : [hidden_dim, T]
    //    Output: [vocab_size, T]
    // ===========================
    Tensor logits(vocab_size, T);
    Tensor col_in(hidden_dim, 1);

    for (int t = 0; t < T; ++t) {
        // extract column t → col_in
        for (int hdim = 0; hdim < hidden_dim; ++hdim) {
            col_in(hdim, 0) = h(hdim, t);
        }

        // apply linear
        Tensor col_out = lm_head.forward(col_in);  // [vocab_size, 1]

        // write back
        for (int v = 0; v < vocab_size; ++v) {
            logits(v, t) = col_out(v, 0);
        }
    }

    info.logits = logits;
    return info;
}
Tensor TransformerModel::forward_incremental(const std::vector<int>& tokens, std::vector<KVCache>& kv_caches) const {
    if (static_cast<int>(kv_caches.size()) != num_layers) {
        kv_caches.assign(static_cast<std::size_t>(num_layers), KVCache{});
    }

    int T = static_cast<int>(tokens.size());
    assert(T > 0);
    int threads = autotune_threads(T);
    omp_set_num_threads(threads);
    Tensor logits(vocab_size, T);

    for (int t = 0; t < T; ++t) {
        int id = tokens[t];
        assert(id >= 0 && id < vocab_size);

        // 1) 只处理当前 token 的 embedding
        Tensor x(hidden_dim, 1);
        for (int h = 0; h < hidden_dim; ++h) {
            x(h, 0) = embedding(id, h);
        }

        // 2) 依次通过每一层的增量前向，复用 KV cache
        Tensor h_col = x;
        for (int layer = 0; layer < num_layers; ++layer) {
            h_col = blocks[layer].forward_incremental(h_col, kv_caches[layer]);
        }

        // 3) 输出层只对当前 token 做一次投影
        Tensor logit_col = lm_head.forward(h_col);  // [vocab_size,1]
        for (int v = 0; v < vocab_size; ++v) {
            logits(v, t) = logit_col(v, 0);
        }
    }

    return logits;
}