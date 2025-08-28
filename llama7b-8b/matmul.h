#pragma once
#include <cstring>

// Matrix multiplication (quantized)
template <int N, int D>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
    static int8_t x_buffer[N];
    static float xs_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = x_buffer type = cyclic factor = 16
#pragma HLS ARRAY_PARTITION variable = xs_buffer type = cyclic factor = 4

x_buff:
    for (int i = 0; i < N; i++) {
#pragma HLS UNROLL factor = 16
        x_buffer[i] = xq[i];
    }

xs_buff:
    for (int j = 0; j <= N - GS; j += GS) {
#pragma HLS UNROLL factor = 4
        xs_buffer[j / GS] = xs[j / GS];
    }
    int i;
    for (i = 0; i < D; i++) {
#pragma HLS PIPELINE
        float val = 0.0f;
        int8_t w_buffer[N];
        float ws_buffer[N / GS];
#pragma HLS ARRAY_PARTITION variable = w_buffer type = cyclic factor = 32
#pragma HLS ARRAY_PARTITION variable = ws_buffer type = cyclic factor = 32
        const int in = i * N;

    matmul1:
        for (int j = 0; j < N; j++) {
            w_buffer[j] = wq[j + in];
        }

    matmul2:
        const int in_s = i * N / GS;
        const int groups = N / GS;
        for (int j = 0; j < groups; j++) {
            ws_buffer[j] = ws[in_s + j];
        }
        int j;

    matmul3:
        for (j = 0; j <= N - GS; j += GS) {
            int32_t ival = 0;

        matmul4:
            for (int k = 0; k < GS; k++) {
                ival += ((int32_t)x_buffer[j + k]) * ((int32_t)w_buffer[j + k]);
            }
            val += ((float)ival) * ws_buffer[j / GS] * xs_buffer[j / GS];
        }
        xout[i] = val;
    }
}
