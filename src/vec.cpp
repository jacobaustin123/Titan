//
//  vec.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include "vec.hpp"
#include <iostream>
#include <cmath>

Vec::Vec() {
    data[0] = 0;
    data[1] = 0;
    data[2] = 0;
}

Vec::Vec(double x, double y, double z) {
    data[0] = x;
    data[1] = y;
    data[2] = z;
}

Vec::Vec(const Vec & v) {
    data[0] = v.data[0];
    data[1] = v.data[1];
    data[2] = v.data[2];
}

Vec Vec::operator+(const Vec & x) {
    return Vec(data[0] + x.data[0], data[1] + x.data[1], data[2] + x.data[2]);
}

Vec Vec::operator-(const Vec & x) {
    return Vec(data[0] - x.data[0], data[1] - x.data[1], data[2] - x.data[2]);
}

Vec Vec::operator/(const Vec & x) {
    return Vec(data[0] / x.data[0], data[1] / x.data[1], data[2] / x.data[2]);
}

Vec Vec::operator*(const double x) {
    return Vec(data[0] * x, data[1] * x, data[2] * x);
}

Vec Vec::operator/(const double x) {
    return Vec(data[0] / x, data[1] / x, data[2] / x);
}

Vec Vec::operator-() {
    return Vec(-data[0], -data[1], -data[2]);
}


Vec operator*(const double x, Vec v) {
    return Vec(v.data[0] * x, v.data[1] * x, v.data[2] * x);
}

Vec operator/(const double x, Vec v) {
    return Vec(v.data[0] / x, v.data[1] / x, v.data[2] / x);
}

std::ostream & operator << (std::ostream & strm, Vec v) {
    return strm << "(" << v[0] << ", " << v[1] << ", " << v[2] << ")";
}

double & Vec::operator [] (int n) {
    if (n < 0 || n >= 3) {
        std::cerr << std::endl << "Out of bounds" << std::endl;
        exit(EXIT_FAILURE);
    } else {
        return data[n];
    }
}

double Vec::norm() {
    return sqrt(pow(data[0], 2) + pow(data[1], 2) + pow(data[2], 2));
}

