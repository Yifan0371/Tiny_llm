//
// Created by liyif on 2025/11/23.
//
#include "layernorm.h"
#include <cmath>
#include <cassert>
#include "weight_loader.h"
LayerNorm::LayerNorm(int dim):gamma(dim,1),beta(dim,1){
    for(int i=0;i<dim;i++){
        gamma(i,0)=1.0f;
        beta(i,0)=0.0f;
    }
}
Tensor LayerNorm::forward(const Tensor& x) const {
    int dim = x.rows();
    int seq_len = x.cols();

    Tensor y(dim, seq_len);

    for (int t = 0; t < seq_len; ++t) {

        // 1. compute mean
        float mean = 0.0f;
        for (int i = 0; i < dim; ++i) {
            mean += x(i, t);
        }
        mean /= dim;

        // 2. compute variance
        float var = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float d = x(i, t) - mean;
            var += d * d;
        }
        var /= dim;

        float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        // 3. normalize and apply gamma/beta
        for (int i = 0; i < dim; ++i) {
            float norm = (x(i, t) - mean) * inv_std;
            y(i, t) = norm * gamma(i, 0) + beta(i, 0);
        }
    }

    return y;
}
void LayerNorm::load_from(WeightLoader& loader) {
    loader.read_into(gamma);
    loader.read_into(beta);
}