#include "utils.h"
#include <iostream>
//using namespace std;
int main() {
    Tensor x(3,1);
    x(0,0) = 1.0;
    x(1,0) = 2.0;
    x(2,0) = 3.0;

    Tensor y = softmax(x);

    std::cout << y(0,0) << "\n";
    std::cout << y(1,0) << "\n";
    std::cout << y(2,0) << "\n";

    return 0;
}
