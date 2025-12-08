#include "model.h"
#include "utils.h"
#include <iostream>
#include <vector>

int main() {

	set_omp_threads(4);
    int vocab = 1000;
    int hidden = 128;
    int num_heads = 8;
    int layers = 2;
    int seq = 16;

    TransformerModel model(vocab, hidden, num_heads, layers, seq);

    std::vector<int> tokens = {10, 20, 30};

    Tensor logits = model.forward(tokens);

    std::cout << "logits shape: "
              << logits.rows() << " × " << logits.cols() << std::endl;

    std::cout << "logits[0][0..5]: ";
    for (int i = 0; i < 6; i++)
        std::cout << logits.data()[i] << " ";
    std::cout << std::endl;

    return 0;
}
