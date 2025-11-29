#include <iostream>
#include "ffn.h"

int main() {
    int hidden_dim = 4;
    int intermediate_dim = 16;

    FFN ffn(hidden_dim, intermediate_dim);

    // 初始化权重
    ffn.fc1.weight.fill(0.1f);
    ffn.fc2.weight.fill(0.1f);

    ffn.fc1.bias.fill(0.0f);
    ffn.fc2.bias.fill(0.0f);

    // 输入
    Tensor x(hidden_dim, 1);
    x(0,0)=1; x(1,0)=2; x(2,0)=3; x(3,0)=4;

    Tensor y = ffn.forward(x);

    std::cout << "FFN output:\n";
    for (int i = 0; i < hidden_dim; ++i) {
        std::cout << y(i,0) << "\n";
    }

    return 0;
}
