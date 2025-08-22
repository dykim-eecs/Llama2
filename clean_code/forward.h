#include "typedefs.h"
#include "config.h"
#include <math.h>
#include <cstring>

extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
                        int token, 
                        int pos, 
                        float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float out[vocab_size]);  // Forward pass declaration

template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {  // Dequantize tensor
    for (int i = 0; i < S; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS) {  // Quantize tensor
    constexpr int num_groups = S / 64;
    constexpr float Q_MAX = 127.0f;
    float scale_buffer[num_groups];
    int8_t quantized_buffer[S];


main_loop:
    for (int group = 0; group < num_groups; group++) {

        float wmax = 0.0;
        int base_idx = group * GS;

    max:
        for (int i = 0; i < GS; i++) {

            float val = fabs(x[base_idx + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        float scale = wmax / Q_MAX;
        scale_buffer[group] = scale;

        for (int i = 0; i < GS; i++) {

            float quant_value = x[base_idx + i] / scale;
            int8_t quantized = (int8_t)round(quant_value);
            quantized_buffer[base_idx + i] = quantized;
        }
    }

    std::memcpy(qx->q, quantized_buffer, S * sizeof(int8_t));
    std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float));
}
