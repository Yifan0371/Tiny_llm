#include "attention.h"
#include "utils.h"   // softmax
#include <cmath>
#include <cassert>
#include <cmath>
#include <omp.h>
#include "weight_loader.h"
Attention::Attention(int hidden_dim, int num_heads)
    : num_heads(num_heads),
      head_dim(hidden_dim / num_heads),
      q_proj(hidden_dim, hidden_dim),
      k_proj(hidden_dim, hidden_dim),
      v_proj(hidden_dim, hidden_dim),
      o_proj(hidden_dim, hidden_dim)
{
    assert(hidden_dim % num_heads == 0);
}

Tensor Attention::forward(const Tensor& x) const {
    // x: [hidden_dim, seq_len]
    int hidden_dim = x.rows();
    int seq_len    = x.cols();

    assert(hidden_dim == head_dim * num_heads);

    // 1. 计算 Q / K / V，全都是 [hidden_dim, seq_len]
    Tensor Q(hidden_dim, seq_len);
    Tensor K(hidden_dim, seq_len);
    Tensor V(hidden_dim, seq_len);

    // 临时列向量，用于调用 Linear::forward
    Tensor x_col(hidden_dim, 1);

    for (int t = 0; t < seq_len; ++t) {
        // 取出第 t 列，作为一个 [hidden_dim,1] 向量
        for (int r = 0; r < hidden_dim; ++r) {
            x_col(r, 0) = x(r, t);
        }

        Tensor q_col = q_proj.forward(x_col);  // [hidden_dim,1]
        Tensor k_col = k_proj.forward(x_col);
        Tensor v_col = v_proj.forward(x_col);

        // 写回到 Q/K/V 的第 t 列
        for (int r = 0; r < hidden_dim; ++r) {
            Q(r, t) = q_col(r, 0);
            K(r, t) = k_col(r, 0);
            V(r, t) = v_col(r, 0);
        }
    }

    // 2. multi-head 计算，每个 head 单独做 attention
    Tensor heads_out(hidden_dim, seq_len);  // 所有 head 的输出拼在一起

    float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    for (int h = 0; h < num_heads; ++h) {
        int row_offset = h * head_dim;

        // 2.1 取出该 head 对应的 Q_h, K_h, V_h: [head_dim, seq_len]
        Tensor Qh(head_dim, seq_len);
        Tensor Kh(head_dim, seq_len);
        Tensor Vh(head_dim, seq_len);

        for (int r = 0; r < head_dim; ++r) {
            for (int t = 0; t < seq_len; ++t) {
                Qh(r, t) = Q(row_offset + r, t);
                Kh(r, t) = K(row_offset + r, t);
                Vh(r, t) = V(row_offset + r, t);
            }
        }

        // 2.2 scores: [seq_len, seq_len]，第 t 行是当前 token 对所有 token 的打分
        Tensor scores(seq_len, seq_len);
        Tensor Oh(head_dim, seq_len);  // 该 head 的输出

		#pragma omp parallel for
        for (int t = 0; t < seq_len; ++t) {
            // 计算第 t 个 token 对所有 j 的 score[t,j]
            for (int j = 0; j < seq_len; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d) {
                    dot += Qh(d, t) * Kh(d, j);
                }
                scores(t, j) = dot * scale;
            }

            // 2.3 对 score 的第 t 行做 softmax → 得到注意力权重 α_t: [seq_len,1]
            Tensor score_row(seq_len, 1);
            for (int j = 0; j < seq_len; ++j) {
                score_row(j, 0) = scores(t, j);
            }
            Tensor alpha = softmax(score_row);  // [seq_len,1]

            // 2.4 用 α_t 对 V_h 做加权求和 → 得到输出向量 Oh[:,t]
            for (int d = 0; d < head_dim; ++d) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; ++j) {
                    float w = alpha(j, 0);
                    sum += w * Vh(d, j);
                }
                Oh(d, t) = sum;
            }
        }

        // 2.5 把当前 head 的输出写回到总的 heads_out 里
        for (int r = 0; r < head_dim; ++r) {
            for (int t = 0; t < seq_len; ++t) {
                heads_out(row_offset + r, t) = Oh(r, t);
            }
        }
    }

    // 3. 输出投影 out_proj：对每一列做一次 Linear
    Tensor y(hidden_dim, seq_len);
    #pragma omp parallel for

    for (int t = 0; t < seq_len; ++t) {
		Tensor in_col(hidden_dim, 1);
        for (int r = 0; r < hidden_dim; ++r) {
            in_col(r, 0) = heads_out(r, t);
        }
        Tensor out_col = o_proj.forward(in_col);  // [hidden_dim,1]
        for (int r = 0; r < hidden_dim; ++r) {
            y(r, t) = out_col(r, 0);
        }
    }

    return y;
}
void Attention::load_from(WeightLoader& loader) {
    q_proj.load_from(loader);
    k_proj.load_from(loader);
    v_proj.load_from(loader);
    o_proj.load_from(loader);
}