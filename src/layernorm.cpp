//
// Created by liyif on 2025/11/23.
//
#include "layernorm.h"
#include <cmath>
#include <cassert>
LayerNorm::LayerNorm(int dim):gamma(dim,1),beta(dim,1){
    for(int i=0;i<dim;i++){
        gamma(i,0)=1.0f;
        beta(i,0)=0.0f;
    }
}
Tensor LayerNorm::forward(const Tensor& x) const {
    assert(x.cols()==1);

    int dim=x.rows();
    Tensor y(dim,1);

    const float* x_ptr = x.fptr();
    const float* g_ptr = gamma.fptr();
    const float* b_ptr = beta.fptr();
    float* y_ptr = y.fptr();

    //均值
    float mean=0.0f;
    for(int i=0;i<dim;i++){mean+=x_ptr[i];}
    mean/=dim;

    float var=0.0f;
    for(int i=0;i<dim;i++){
        float diff=x_ptr[i]-mean;
        var+=diff*diff;
    }
    var/=dim;

    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for(int i=0;i<dim;i++){
        float norm=(x_ptr[i]-mean)*inv_std;
        y_ptr[i]=norm*g_ptr[i]+b_ptr[i];
    }
    return y;
}