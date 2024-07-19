# include <iostream>
# include "../cTorch/Tensor.hpp"

bool test_addition_operator () {
    const double x = 5.;

    cTorch::Tensor lhs {
        1, 3, 3, 
        std::vector<double>{
            1, 2, 3, 
            4, 5, 6,
            7, 8, 9
        }
    };

    cTorch::Tensor rhs {
        1, 3, 3,
        std::vector<double>{
            6, 7, 8,
            9, 10, 11,
            12, 13, 14
        }
    };

    cTorch::Tensor res {
        1, 3, 3,
        std::vector<double>{
            7, 9, 11,
            13, 15, 17,
            19, 21, 23
        }
    };

    bool result = true;

    result = result && ((lhs + x) == rhs);
    result = result && ((lhs + rhs) == res);

    return result;

}

int main() {
    std::cout << test_addition_operator() << std::endl;
}
