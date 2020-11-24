
#include "dot.h"

float dot_product(float* lhs, float* rhs, int n_elements) {
    float out = 0.0;

#pragma acc kernels
{
    for(int i=0; i < n_elements; i++) {
        out += lhs[i] * rhs[i];
    }

}

    return out;
}
