//
//  vec.cpp
//  CUDA Physics
//
//  Created by Jacob Austin on 5/13/18.
//  Copyright © 2018 Jacob Austin. All rights reserved.
//

#include "vec.h"

using namespace titan;

namespace titan {

#if __CUDA_ARCH__ < 600
__device__ double atomicDoubleAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
            (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

CUDA_DEVICE void Vec::atomicVecAdd(const Vec & v) {
atomicDoubleAdd(&data[0], (double) v.data[0]);
atomicDoubleAdd(&data[1], (double) v.data[1]);
atomicDoubleAdd(&data[2], (double) v.data[2]);
}

CUDA_CALLABLE_MEMBER double dot(const Vec & a, const Vec & b) {
    return (a * b).sum();
}

CUDA_CALLABLE_MEMBER Vec cross(const Vec &v1, const Vec &v2) {
    return Vec(v1[1] * v2[2] - v1[2] * v2[1], v2[0] * v1[2] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0]);
}

}