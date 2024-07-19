# include "Tensor.hpp"

namespace cTorch {

    const uLong Tensor::_idx (const uLong i, const uLong j, const uLong k) const {
        return i*y + j*z + k;
    }

    const uLong Tensor::_idx (const uLong i, const uLong j, const uLong k,
                              const uLong x, const uLong y, const uLong z) const {
        return i*y + j*z + k;
    }

    const double Tensor::_val (const uLong i, const uLong j, const uLong k) const {
        return data[_idx(i,j,k)];
    }

    // Constructors
    Tensor::Tensor (const uLong y, const uLong z) 
    : Tensor (1, y, z, 0.)
    {}

    Tensor::Tensor (const uLong x, const uLong y, const uLong z) 
    : Tensor (x, y, z, 0.)
    {}

    Tensor::Tensor (const uLong y, const uLong z, const double value) 
    : x(1), y(y), z(z) 
    {
        data.assign (x*y*z, value);
        data.shrink_to_fit ();
    }

    Tensor::Tensor (const uLong x, const uLong y, const uLong z, const double value) 
    : x(x), y(y), z(z) 
    {
        data.assign (x*y*z, value);
        data.shrink_to_fit ();
    }

    Tensor::Tensor(const uLong x, const uLong y, const uLong z, const std::vector<double>& data)
    : x(x), y(y), z(z), data(data)
    {
        if (data.size() != x*y*z)
            std::length_error ("The values x * y * z must match data size");

        this->data.shrink_to_fit ();
    }

    // Tensor::Tensor (uLong x, uLong y, uLong z, double* values)
    // : x(x), y(y), z(z), data(values, values + unsigned(sizeof(values) / sizeof(double)))
    // {
    //     if (data.size() != unsigned(x*y*z))
    //         std::length_error ("The values x * y * z must match data size");

    //     data.shrink_to_fit ();
    // }

    // Copy Constructot and Copy Assignment Operator

    Tensor::Tensor(const Tensor& other)
    : x(other.x), y(other.y), z(other.z)
    {
        data.assign(other.data.begin(), other.data.end());
    }

    Tensor& Tensor::operator=(const Tensor& other) 
    {
        return *this = Tensor(other);
    }

    // Move Constructor and Move Assignment Operator

    Tensor::Tensor (Tensor&& other) noexcept
    : x(other.x), y(other.y), z(other.z), data(std::move(other.data)) 
    {}

    Tensor& Tensor::operator=(Tensor&& other)  noexcept
    {
        std::swap(*this, other);
        return *this;
    }

    // Print operator overload
    std::ostream& operator<<(std::ostream& os, const Tensor& T)
    {
        os <<   "x: " << T.x 
           << "\ty: " << T.y
           << "\tz: " << T.z;

        os << "\n";

        // print the array
        os << "[\n";
        for (uLong i = 0; i < T.x; i++) {
            os << "\t[\n";
            for (uLong j = 0; j < T.y; j++) {
                os << '\t';
                for (uLong k = 0; k <  T.z; k++) {
                    os << "\t" << T._val(i, j, k);
                }
                os << "\n";
            }
            os << "\t]\n";
        }
        os << "]" << std::endl;
        return os;
    }

    // size of vector
    uLong Tensor::size() const { return data.size(); }

    // Overload [] operator
    double& Tensor::operator[] (const uLong i) {
        return data[i];
    }

    // Arthmetic operators overload
    Tensor& Tensor::operator+= (const double other) {
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            *itr += other;
        }
        return *this;
    }

    Tensor& Tensor::operator-= (const double other) {
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            *itr -= other;
        }
        return *this;
    }

    Tensor& Tensor::operator*= (const double other) {
        for (auto itr = data.begin(); itr != data.end(); itr++) {
            *itr *= other;
        }
        return *this;
    }

    Tensor& Tensor::operator/= (const double other) {
        if (!other) std::logic_error("Division by zero!");
        auto inv = 1 / other;
        *this *= inv;
        return *this;
    }

    Tensor& Tensor::operator+= (const Tensor& other) {
        if (
            data.size() == other.data.size() // Tensor - Tensor addition
        ||    y*z == other.data.size() // Tensor - Matrix addition
        ||  z == other.data.size() // Tensor - Vector addition
        ) {
            // initialise
            auto itr1 = data.begin();
            auto itr2 = other.data.begin();
            while (itr1 != data.end()) {
                // add 
                *itr1 += *itr2;
                // update
                itr1++;
                itr2++;
                // reset the iterator
                if (itr2 == other.data.end()) 
                    itr2 = other.data.begin();
            }
        } else {
            std::length_error ("The size of both vector are not same.");
        }
        return *this;
    }

    Tensor& Tensor::operator-= (const Tensor& other) {
        *this += Tensor(other) * -1.; // using copy constructor
        return *this;
    }

    Tensor operator+ (const Tensor& lhs, const double rhs) {
        auto new_tensor {lhs};
        new_tensor += rhs;
        return new_tensor;
    }

    Tensor operator- (const Tensor& lhs, const double rhs) {
        auto new_tensor {lhs};
        new_tensor -= rhs;
        return new_tensor;
    }

    Tensor operator* (const Tensor& lhs, const double rhs) {
        auto new_tensor {lhs};
        new_tensor *= rhs;
        return new_tensor;
    }

    Tensor operator/ (const Tensor& lhs, const double rhs) {
        auto new_tensor {lhs};
        new_tensor /= rhs;
        return new_tensor;
    }

    Tensor operator+ (const Tensor& lhs, const Tensor& rhs){
        auto new_tensor {lhs};
        new_tensor += rhs;
        return new_tensor;
    }

    Tensor operator- (const Tensor& lhs, const Tensor& rhs){
        auto new_tensor {lhs};
        new_tensor -= rhs;
        return new_tensor;
    }

    Tensor operator* (const Tensor& lhs, const Tensor& rhs){

        if (lhs.z != rhs.y || ~(lhs.x == rhs.x || rhs.x == 1)) {
            std::length_error ("The shape of both vector do not match");
            return lhs;
        }

        auto res = Tensor(lhs.x, lhs.y, rhs.z);

        for (auto i = 0, ii = 0; i < lhs.x; ++i, ++ii) {
            for (auto j = 0; j < lhs.y; ++j) {
                for (auto k = 0; k < rhs.z; ++k) {
                    double sum {};
                    for (auto l = 0; l < lhs.z; ++l) {
                        sum += lhs.data[lhs._idx(i, j, l)] * rhs.data[rhs._idx(ii, l, k)];
                    }
                    res[res._idx(i, j, k)] = sum;
                }
            }
            if (rhs.x == 1) ii = 0;
        }

        return res;

    }


    // Transpose
    Tensor& Tensor::T() {
        // Exchange rows with columns
        for (auto i = 0; i < x; i++)
            for (auto j = 0; j < y; j++)
                for (auto k = j; k < z; k++) {
                    std::swap(data[_idx(i, j, k)], data[_idx(i, k, j, x, z, y)]);
                    // printf("swapping: %d, %d, %d (%d) -> (%d)\n", i, j, k, _idx(i, j, k), _idx(i, k, j, x, z, y));
                }
        
        std::swap(y, z);
        return *this;
    }

    // overload == operator
    const bool Tensor::operator== (const Tensor& other) const {
        return data == other.data; 
    }


} // end cTorch namespace