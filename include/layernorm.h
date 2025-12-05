//
// Created by liyif on 2025/11/23.
//

#pragma once
#include "tensor.h"


class WeightLoader;



class LayerNorm{
public:
    Tensor gamma;
    Tensor beta;

    LayerNorm(int dim);

    Tensor forward(const Tensor& x) const;

	    // Load gamma and beta values from a shared binary weight stream.
    void load_from(WeightLoader& loader);
};