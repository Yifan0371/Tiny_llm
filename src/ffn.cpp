//
// Created by liyif on 2025/11/29.
//
#include "ffn.h"
#include <cassert>
#include "weight_loader.h"
FFN::FFN(int hidden_dim, int intermediate_dim):fc1(intermediate_dim, hidden_dim),fc2(hidden_dim,intermediate_dim)
{
}

Tensor FFN::forward(const Tensor &x)const
{
    int hidden_dim=x.rows();//行数
    int seq_len=x.cols();//列数

    //创建输出张量
    Tensor y(hidden_dim,seq_len);
    Tensor x_col(hidden_dim, 1);

    for(int t=0;t<seq_len;t++){
        for(int r=0;r<hidden_dim;r++){
            x_col(r,0)=x(r,t);
        }
        // Step 2: FC1 → GELU
        Tensor h1=fc1.forward(x_col);
        Tensor h1_gelu=gelu(h1);

        //fc2
        Tensor h2=fc2.forward(h1_gelu);
        for(int r=0;r<hidden_dim;r++){
            y(r,t)=h2(r,0);
        }
    }
    return y;
}

void FFN::load_from(WeightLoader& loader){
    fc1.load_from(loader);
    fc2.load_from(loader);
}



