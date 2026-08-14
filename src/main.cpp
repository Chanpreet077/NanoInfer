#include "tensor.h"
#include <iostream>

int main()
{
    Tensor a({2, 2});
    Tensor b({2, 2});

    a.at(0, 0) = 1;
    a.at(0, 1) = 2;
    a.at(1, 0) = 3;
    a.at(1, 1) = 4;

    b.at(0, 0) = 5;
    b.at(0, 1) = 6;
    b.at(1, 0) = 7;
    b.at(1, 1) = 8;

    Tensor c = a.matmul(b);

    std::cout << "A x B: ";
    c.print();

    return 0;
}