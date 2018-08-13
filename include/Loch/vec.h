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

#ifdef __CUDACC__
#define CUDA_DEVICE __device__
#else
#define CUDA_DEVICE
#endif

#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <cmath>

class Vec {
public:
    CUDA_CALLABLE_MEMBER Vec() {
        data[0] = 0;
        data[1] = 0;
        data[2] = 0;
    } // default

    CUDA_CALLABLE_MEMBER Vec(const Vec & v) {
        data[0] = v.data[0];
        data[1] = v.data[1];
        data[2] = v.data[2];
    } // copy constructor

    CUDA_CALLABLE_MEMBER Vec(double x, double y, double z) {
        data[0] = x;
        data[1] = y;
        data[2] = z;
    } // initialization from x, y, and z values

    CUDA_CALLABLE_MEMBER Vec & operator=(const Vec & v) {
        if (this == &v) {
            return *this;
        }

        data[0] = v.data[0];
        data[1] = v.data[1];
        data[2] = v.data[2];

        return *this;
    }

    CUDA_CALLABLE_MEMBER Vec & operator+=(const Vec & v) {
        data[0] += v.data[0];
        data[1] += v.data[1];
        data[2] += v.data[2];
        return *this;
    }

    CUDA_CALLABLE_MEMBER Vec & operator-=(const Vec & v) {
        data[0] -= v.data[0];
        data[1] -= v.data[1];
        data[2] -= v.data[2];
        return *this;
    }

    CUDA_DEVICE void atomicVecAdd(const Vec & v);

    CUDA_CALLABLE_MEMBER Vec operator-() const{
        return Vec(-data[0], -data[1], -data[2]);
    }

    CUDA_CALLABLE_MEMBER double & operator [] (int n) {
        if (n < 0 || n >= 3) {
            printf("%s\n", "Out of bounds!");
            return data[0];
        } else {
            return data[n];
        }
    }

    CUDA_CALLABLE_MEMBER const double & operator [] (int n) const {
        if (n < 0 || n >= 3) {
            printf("%s\n", "Out of bounds!");
            return data[0];
        } else {
            return data[n];
        }
    }

    CUDA_CALLABLE_MEMBER friend Vec operator+(const Vec & v1, const Vec & v2) {
        return Vec(v1.data[0] + v2.data[0], v1.data[1] + v2.data[1], v1.data[2] + v2.data[2]);
    }

    CUDA_CALLABLE_MEMBER friend Vec operator-(const Vec & v1, const Vec & v2) {
        return Vec(v1.data[0] - v2.data[0], v1.data[1] - v2.data[1], v1.data[2] - v2.data[2]);
    }

    CUDA_CALLABLE_MEMBER friend Vec operator*(const double x, const Vec & v) {
        return Vec(v.data[0] * x, v.data[1] * x, v.data[2] * x);
    }

    CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec & v, const double x) {
        return x * v;
    } // double times Vec

    CUDA_CALLABLE_MEMBER friend bool operator==(const Vec & v1, const Vec & v2) {
        return (v1[0] == v2[0] && v1[1] == v2[1] && v1[2] == v2[2]);
    }

    CUDA_CALLABLE_MEMBER friend Vec operator*(const Vec & v1, const Vec & v2) {
        return Vec(v1.data[0] * v2.data[0], v1.data[1] * v2.data[1], v1.data[2] * v2.data[2]);
    } // Multiplies two Vecs (elementwise)

    CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec & v, const double x) {
        return Vec(v.data[0] / x, v.data[1] / x, v.data[2] / x);
    } //  vector over double

    CUDA_CALLABLE_MEMBER friend Vec operator/(const Vec & v1, const Vec & v2) {
        return Vec(v1.data[0] / v2.data[0], v1.data[1] / v2.data[1], v1.data[2] / v2.data[2]);
    } // divides two Vecs (elementwise)

    friend std::ostream & operator << (std::ostream & strm, const Vec & v) {
        return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
    } // print

    CUDA_CALLABLE_MEMBER void print() {
        printf("(%3f, %3f, %3f)\n", data[0], data[1], data[2]);
    }

    CUDA_CALLABLE_MEMBER double norm() const {
        return sqrt(pow(data[0], 2) + pow(data[1], 2) + pow(data[2], 2));
    } // gives vector norm

    CUDA_CALLABLE_MEMBER double sum() const {
        return data[0] + data[1] + data[2];
    } // sums all components of the vector

private:
    double data[3] = { 0 }; // initialize data to 0
};

CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b);
CUDA_CALLABLE_MEMBER Vec cross(const Vec &v1, const Vec &v2);

#endif

