#ifndef OPTIMIZED_RMSNORM_H
#define OPTIMIZED_RMSNORM_H

#include <cmath>
#include <ap_int.h>
#include <hls_stream.h>


template <int S>
void sum_of_squares_proc(float x_in[S], hls::stream<float>& x_stream, hls::stream<float>& ss_stream) {
    float ss_local = 0.0f;
    for (int j = 0; j < S; j++) {
#pragma HLS pipeline II=1
        float x_val = x_in[j];
        x_stream.write(x_val);
        ss_local += x_val * x_val;
    }
    ss_stream.write(ss_local);
}


template <int S>
void normalize_and_scale_proc(hls::stream<float>& ss_stream, hls::stream<float>& x_stream, float weight_in[S], float o_out[S]) {
    float ss_val = ss_stream.read();
    float ss_scaled = 1.0f / sqrtf(ss_val / S + 1e-5f);
    
    for (int j = 0; j < S; j++) {
#pragma HLS pipeline II=1
        float x_val = x_stream.read();
        float weight_val = weight_in[j];
        o_out[j] = weight_val * (ss_scaled * x_val);
    }
}



template <int S>
void rmsnorm(float o[S], float x[S], float weight[S]) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=768
#pragma HLS INTERFACE m_axi port=o offset=slave bundle=gmem1 depth=768
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2 depth=768
#pragma HLS dataflow

    // Stream for transferring input values from sum_of_squares_proc to normalize_and_scale_proc.
    hls::stream<float> x_stream;
#pragma HLS STREAM variable=x_stream depth=768

    // New stream for transferring the single sum_of_squares value.
    hls::stream<float> ss_stream;
#pragma HLS STREAM variable=ss_stream depth=1

    // Process 1: Calculate sum of squares and stream input values.
    sum_of_squares_proc<S>(x, x_stream, ss_stream);

    // Process 2: Read sum of squares and input values from streams, then normalize and scale.
    normalize_and_scale_proc<S>(ss_stream, x_stream, weight, o);
}

#endif
