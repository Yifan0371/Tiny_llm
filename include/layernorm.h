//
// Created by liyif on 2025/11/23.
//

#pragma once
#include "tensor.h"

class LayerNorm{
public:
    Tensor gamma;
    Tensor beta;

    LayerNorm(int dim);

    Tensor forward(const Tensor& x) const;
};