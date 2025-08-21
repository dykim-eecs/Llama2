#include <cmath>
#include <ap_int.h>
#include <hls_stream.h>

#define S 768

// HLS_DATAFLOW_KANONICAL_FIXED
void sum_of_squares_proc(float x_in[S], hls::stream<float>& x_stream, float* ss_out) {
    float ss_local = 0.0f;
    for (int j = 0; j < S; j++) {
#pragma HLS pipeline II=1
        float x_val = x_in[j];
        x_stream.write(x_val);
        ss_local += x_val * x_val;
    }
    *ss_out = ss_local;
}

void normalize_and_scale_proc(float ss_val, hls::stream<float>& x_stream, float weight_in[S], float o_out[S]) {
    float ss_scaled = 1.0f / sqrtf(ss_val / S + 1e-5f);
    
    for (int j = 0; j < S; j++) {
#pragma HLS pipeline II=1
        float x_val = x_stream.read();
        float weight_val = weight_in[j];
        o_out[j] = weight_val * (ss_scaled * x_val);
    }
}


// RMS normalization
extern "C" void rmsnorm(float o[S], float x[S], float weight[S]) {
#pragma HLS INTERFACE m_axi port=x offset=slave bundle=gmem0 depth=768
#pragma HLS INTERFACE m_axi port=o offset=slave bundle=gmem1 depth=768
#pragma HLS INTERFACE m_axi port=weight offset=slave bundle=gmem2 depth=768
#pragma HLS dataflow

    hls::stream<float> x_stream;
#pragma HLS STREAM variable=x_stream depth=768

    float ss_val;

    sum_of_squares_proc(x, x_stream, &ss_val);

    normalize_and_scale_proc(ss_val, x_stream, weight, o);
}
