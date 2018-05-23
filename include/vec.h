//
//  vec.hpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#ifndef VEC_H
#define VEC_H

#include <iostream>

class Vec {
public:
    Vec(); // default
    Vec(const Vec & v); // copy constructor
    Vec(double x, double y, double z); // initialization from x, y, and z values
    Vec & operator = (const Vec & v);
    Vec & operator += (const Vec & v);
    Vec operator - () const;
    double & operator [] (int n);
    const double & operator [] (int n) const;

    friend Vec operator + (const Vec & x, const Vec & y);
    friend Vec operator - (const Vec & x, const Vec & y);

    friend Vec operator * (const Vec & v, const double x); // double and Vec
    friend Vec operator * (const double x, const Vec & v);
    friend Vec operator * (const Vec & v1, const Vec & v2); // two Vecs (elementwise)

    friend Vec operator / (const Vec & v, const double x); // double and vec
//    friend Vec operator / (const double x, const Vec & v); // not needed
    friend Vec operator / (const Vec & v1, const Vec & v2); // two Vecs (elementwise)

    friend std::ostream & operator << (std::ostream &, const Vec &); // print
    
    double norm() const; // gives vector norm
    double sum() const; // gives vector norm

private:
    double data[3] = { 0 }; // initialize data to 0
};

double dot(const Vec & a, const Vec & b);

#endif

