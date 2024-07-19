# pragma once

# include <algorithm>
# include <vector>
# include <ostream>
# include <stdexcept>
# include <utility>

typedef unsigned long uLong;

namespace cTorch {

    class Tensor {
        private:
        uLong x {0}, y {0}, z {0} ;
        std::vector<double> data {} ;


        const uLong _idx (const uLong, const uLong, const uLong) const ; 
        const uLong _idx (const uLong, const uLong, const uLong, 
                          const uLong, const uLong, const uLong) const ;
        const double _val (const uLong, const uLong, const uLong) const  ;

        public:
        // Default Constructor and Destructor
        Tensor() = default ;
        ~Tensor() = default ;

        // Constructors
        Tensor(const uLong, const uLong) ;
        Tensor(const uLong, const uLong, const uLong) ;
        Tensor(const uLong, const uLong, const double) ;
        Tensor(const uLong, const uLong, const uLong, const double) ;
        Tensor(const uLong, const uLong, const uLong, const std::vector<double>&) ;
        // Tensor(uLong, uLong, uLong, double*);

        // Copy Constructot and Copy Assignment Operator 
        Tensor(const Tensor&);
        Tensor& operator=(const Tensor&);

        // Move Constructor and Move Assignment Operator
        Tensor(Tensor&&) noexcept ;
        Tensor& operator=(Tensor&&) noexcept ;

        // getters and Setters
        uLong& get_x() { return x; }
        uLong& get_y() { return y; }
        uLong& get_z() { return z; }
        std::vector<double>& get_vector() { return data; }
        double& get(uLong i, uLong j, uLong k) 
        {return data[_idx(i, j, k)];}

        // Print operator overload
        friend std::ostream& operator<<(std::ostream&, const Tensor&) ;

        // size of vector
        uLong size() const ;

        // Overload [] operator
        double& operator[] (const uLong) ;

        // Arthmetic operators overload
        Tensor& operator+= (const double) ;
        Tensor& operator-= (const double) ;
        Tensor& operator*= (const double) ;
        Tensor& operator/= (const double) ;

        Tensor& operator+= (const Tensor&) ;
        Tensor& operator-= (const Tensor&) ;
        Tensor& operator*= (const Tensor&) = delete ;
        Tensor& operator/= (const Tensor&) = delete ;

        friend Tensor operator+ (const Tensor&, const double) ;
        friend Tensor operator- (const Tensor&, const double) ;
        friend Tensor operator* (const Tensor&, const double) ;
        friend Tensor operator/ (const Tensor&, const double) ;

        friend Tensor operator+ (const Tensor&, const Tensor&) ;
        friend Tensor operator- (const Tensor&, const Tensor&) ;
        friend Tensor operator* (const Tensor&, const Tensor&) ;
        friend Tensor operator/ (const Tensor&, const Tensor&) = delete ;

        // Transpose 
        Tensor& T() ;

        // overload == operator
        const bool operator== (const Tensor&) const ;


        // TODO: matrix inverse
    };

}