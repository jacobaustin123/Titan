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
    double data[3] = { 0 };
    Vec();
    Vec(const Vec & v);
    Vec(double x, double y, double z);
    
    Vec operator + (const Vec & x);
    Vec operator - (const Vec & x);
    Vec operator / (const Vec & x);
    Vec operator * (const double x);
    Vec operator / (const double x);
    Vec operator - ();
    double & operator [] (int n);
    
    friend Vec operator * (const double x, Vec v);
    friend Vec operator / (const double x, Vec v);
    friend std::ostream & operator << (std::ostream &, Vec);
    
    double norm();
};

#endif

