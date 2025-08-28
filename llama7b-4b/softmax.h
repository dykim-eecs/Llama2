#pragma once
#include <cstring>

// Softmax activation
template <int MAXSIZE>
void softmax(float *x, int size) {
    float buffer[MAXSIZE];
    float max_val = x[0];

max:
    for (int i = 1; i < size; i++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129

        float x_i = x[i];
        if (x_i > max_val) {
            max_val = x_i;
        }
    }

exp:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129

        float x_i = expf(x[i] - max_val);
        buffer[i] = x_i;
    }
    float sum = 0.0f;

sum:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
        sum += buffer[i];
    }
    const float inv_sum = 1.0 / sum;

norm:
    for (int i = 0; i < size; i++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129

        x[i] = buffer[i] * inv_sum;
    }
}
