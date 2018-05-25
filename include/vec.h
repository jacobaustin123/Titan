//
//  vec.hpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#ifndef VEC_H
#define VEC_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>

class Vec {
public:
    CUDA_CALLABLE_MEMBER Vec(); // default
    CUDA_CALLABLE_MEMBER Vec(const Vec & v); // copy constructor
    CUDA_CALLABLE_MEMBER Vec(double x, double y, double z); // initialization from x, y, and z values
    CUDA_CALLABLE_MEMBER Vec & operator = (const Vec & v);
    CUDA_CALLABLE_MEMBER Vec & operator += (const Vec & v);
    CUDA_CALLABLE_MEMBER Vec operator - () const;
    CUDA_CALLABLE_MEMBER double & operator [] (int n);
    CUDA_CALLABLE_MEMBER const double & operator [] (int n) const;

    CUDA_CALLABLE_MEMBER friend Vec operator + (const Vec & x, const Vec & y);
    CUDA_CALLABLE_MEMBER friend Vec operator - (const Vec & x, const Vec & y);

    CUDA_CALLABLE_MEMBER friend Vec operator * (const Vec & v, const double x); // double and Vec
    CUDA_CALLABLE_MEMBER friend Vec operator * (const double x, const Vec & v);
    CUDA_CALLABLE_MEMBER friend Vec operator * (const Vec & v1, const Vec & v2); // two Vecs (elementwise)

    CUDA_CALLABLE_MEMBER friend Vec operator / (const Vec & v, const double x); // double and vec
//    friend Vec operator / (const double x, const Vec & v); // not needed
    CUDA_CALLABLE_MEMBER friend Vec operator / (const Vec & v1, const Vec & v2); // two Vecs (elementwise)

    CUDA_CALLABLE_MEMBER friend std::ostream & operator << (std::ostream &, const Vec &); // print

    CUDA_CALLABLE_MEMBER double norm() const; // gives vector norm
    CUDA_CALLABLE_MEMBER double sum() const; // gives vector norm

private:
    double data[3] = { 0 }; // initialize data to 0
};

double dot(const Vec & a, const Vec & b);

#endif

