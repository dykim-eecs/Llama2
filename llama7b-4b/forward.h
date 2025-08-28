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

// ----------------------------------------------------------------------------
// Dequantization for 4-bit
template <int S>
void dequantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    // GS를 템플릿 인자가 아닌 상수로 고정
    constexpr int GS_const = 64;
    for (int i = 0; i < S; i++) {
        int8_t quantized_value;
        // 짝수 인덱스는 하위 4비트, 홀수 인덱스는 상위 4비트
        if (i % 2 == 0) {
            quantized_value = qx->q[i / 2] & 0x0F;
            // 4비트 부호 확장 (sign extension)
            if (quantized_value & 0x08) { // 최상위 비트가 1이면 음수
                quantized_value |= 0xF0;
            }
        } else {
            quantized_value = (qx->q[i / 2] >> 4) & 0x0F;
            if (quantized_value & 0x08) {
                quantized_value |= 0xF0;
            }
        }
        x[i] = static_cast<float>(quantized_value) * qx->s[i / GS_const];
    }
}

// ----------------------------------------------------------------------------
// Quantization for 4-bit
template <int S>
void quantize(QuantizedTensor<S> *qx, float x[S], int GS) {
    // GS를 템플릿 인자가 아닌 상수로 고정하여 컴파일 에러 해결
    constexpr int GS_const = 64;
    constexpr int num_groups = S / GS_const;
    // 4비트 부호 있는 정수의 최대값은 7입니다.
    constexpr float Q_MAX = 7.0f;
    float scale_buffer[num_groups];
    // 4비트 값 두 개를 8비트 정수 하나에 담기 때문에 크기가 절반
    int8_t quantized_buffer[S / 2];

    // 버퍼를 0으로 초기화
    memset(quantized_buffer, 0, sizeof(quantized_buffer)); 

main_loop:
    for (int group = 0; group < num_groups; group++) {
        float wmax = 0.0;
        int base_idx = group * GS_const;

    max:
        for (int i = 0; i < GS_const; i++) {
            float val = fabs(x[base_idx + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        float scale = wmax / Q_MAX;
        scale_buffer[group] = scale;

        for (int i = 0; i < GS_const; i++) {
            float quant_value = x[base_idx + i] / scale;
            int8_t quantized = static_cast<int8_t>(round(quant_value));

            // 값을 -8 ~ 7 범위로 클리핑
            if (quantized > 7) {
                quantized = 7;
            } else if (quantized < -8) {
                quantized = -8;
            }
            
            // 4비트 값을 8비트 버퍼에 압축(packing)하여 저장
            int buffer_idx = (base_idx + i) / 2;
            if ((base_idx + i) % 2 == 0) { // 짝수 인덱스: 하위 4비트에 저장
                quantized_buffer[buffer_idx] |= (quantized & 0x0F); 
            } else { // 홀수 인덱스: 상위 4비트에 저장
                quantized_buffer[buffer_idx] |= ((quantized & 0x0F) << 4);
            }
        }
    }

    std::memcpy(qx->q, quantized_buffer, (S / 2) * sizeof(int8_t));
    std::memcpy(qx->s, scale_buffer, num_groups * sizeof(float));
}