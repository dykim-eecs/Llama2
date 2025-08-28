#include "forward.h"
#include "config.h"
#include <cstring>

#include "rmsnorm.h"
#include "softmax.h"
#include "matmul.h"

// Forward pass
extern "C" void forward(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *transformer, 
                        int token, 
                        int pos, 
                        float key_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float value_cache[n_layers * seq_len * ((dim * n_kv_heads) / n_heads)], 
                        float out[vocab_size]) {
    enum {
      KV_DIM          = (dim * n_kv_heads) / n_heads,
      KV_CACHE_DEPTH  = n_layers * seq_len * KV_DIM,
      OUT_DEPTH       = vocab_size
    };
    
    #pragma HLS INTERFACE mode=m_axi port=transformer bundle=gmem0 offset=slave
    #pragma HLS INTERFACE mode=s_axilite port=token
    #pragma HLS INTERFACE mode=s_axilite port=pos

    #pragma HLS INTERFACE m_axi port=key_cache  bundle=gmem1 offset=slave depth=KV_CACHE_DEPTH
    #pragma HLS INTERFACE m_axi port=value_cache bundle=gmem2 offset=slave depth=KV_CACHE_DEPTH
    #pragma HLS INTERFACE m_axi port=out        bundle=gmem3 offset=slave depth=OUT_DEPTH
    #pragma HLS INTERFACE mode=s_axilite port=return
    
    
    auto w = &transformer->weights;

    static float x[config.dim];
    static float xb[config.dim];
    static float xb2[config.dim];
    static float hb[config.hidden_dim];
    static float hb2[config.hidden_dim];
    static QuantizedTensor<config.dim> xq;
    static QuantizedTensor<config.hidden_dim> hq;
    static float q[config.dim];
    static float k[(config.dim * config.n_kv_heads) / config.n_heads];
    static float v[(config.dim * config.n_kv_heads) / config.n_heads];
    static float att[config.n_heads * config.seq_len];

    constexpr int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    constexpr int kv_mul = config.n_heads / config.n_kv_heads;
    constexpr int head_size = dim / config.n_heads;
    std::memcpy(x, w->token_embedding_table + token * dim, dim * sizeof(float));

main_forward_loop:
    for (int l = 0; l < config.n_layers; l++) {
        rmsnorm<dim>(xb, x, w->rms_att_weight + l * dim);  // Attention RMSNorm
        quantize(&xq, xb, GS);
        matmul<dim, dim>(q, xq.q, xq.s, (w->wq + l)->q, (w->wq + l)->s);  // QKV projections
        matmul<dim, kv_dim>(k, xq.q, xq.s, (w->wk + l)->q, (w->wk + l)->s);
        matmul<dim, kv_dim>(v, xq.q, xq.s, (w->wv + l)->q, (w->wv + l)->s);

    // RoPE for Q and K
    rotation1:
        for (int i = 0; i < kv_dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float v0_q = q[i];
            float v1_q = q[i + 1];
            q[i] = v0_q * fcr - v1_q * fci;
            q[i + 1] = v0_q * fci + v1_q * fcr;
            float v0_k = k[i];
            float v1_k = k[i + 1];
            k[i] = v0_k * fcr - v1_k * fci;
            k[i + 1] = v0_k * fci + v1_k * fcr;
        }

    // RoPE for remaining Q
    rotation2:
        for (int i = kv_dim; i < dim; i += 2) {
            int head_dim = i % head_size;
            float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
            float val = pos * freq;
            float fcr = cosf(val);
            float fci = sinf(val);
            float v0 = q[i];
            float v1 = q[i + 1];
            q[i] = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        int loff = l * config.seq_len * kv_dim;
        float *key_cache_row = key_cache + loff + pos * kv_dim;
        float *value_cache_row = value_cache + loff + pos * kv_dim;
        std::memcpy(key_cache_row, k, kv_dim * sizeof(*key_cache_row));  // Cache K and V
        std::memcpy(value_cache_row, v, kv_dim * sizeof(*value_cache_row));
        int h;

    multihead_attention:
        for (h = 0; h < n_heads; h++) {
            const int q_offset = h * head_size;
            const int att_offset = h * seq_len;

        iterate:
            for (int t = 0; t <= pos; t++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
                const int key_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) {
                    score += q[i + q_offset] * key_cache[i + key_offset];
                }
                score /= sqrtf(head_size);
                att[t + att_offset] = score;
            }

            softmax<257>(att + att_offset, pos + 1);  // Softmax attention scores
            const int xb_offset = h * head_size;
            memset(xb + xb_offset, 0, head_size * sizeof(float));

        acc:
            for (int t = 0; t <= pos; t++) {
#pragma HLS loop_tripcount min = 0 max = 257 avg = 129
                const int v_offset = loff + t * kv_dim + (h / kv_mul) * head_size;
                float a = att[t + att_offset];

            acc_inner:
                for (int i = 0; i < head_size; i++) {
                    xb[i + xb_offset] += a * value_cache[i + v_offset];
                }
            }
        }

        quantize(&xq, xb, GS);
        matmul<dim, dim>(xb2, xq.q, xq.s, (w->wo + l)->q, (w->wo + l)->s);  // Output projection

    residual:
        for (int i = 0; i < dim; i++) {

            x[i] += xb2[i];
        }

        rmsnorm<dim>(xb, x, w->rms_ffn_weight + l * dim);  // FFN RMSNorm
        quantize(&xq, xb, GS);
        matmul<dim, hidden_dim>(hb, xq.q, xq.s, (w->w1 + l)->q, (w->w1 + l)->s);  // FFN projections
        matmul<dim, hidden_dim>(hb2, xq.q, xq.s, (w->w3 + l)->q, (w->w3 + l)->s);

        float hb_out[hidden_dim];


    swi_glu:
        for (int i = 0; i < hidden_dim; i++) {

            float val = hb[i];
            val *= (1.0f / (1.0f + expf(-val)));  // SiLU
            val *= hb2[i];
            hb_out[i] = val;
        }

        std::memcpy(hb, hb_out, hidden_dim * sizeof(float));
        quantize(&hq, hb, GS);
        matmul<hidden_dim, dim>(xb, hq.q, hq.s, (w->w2 + l)->q, (w->w2 + l)->s);  // FFN output

    residual2:
        for (int i = 0; i < dim; i++) {

            x[i] += xb[i];
        }
    }

    rmsnorm<dim>(x, x, w->rms_final_weight);  // Final RMSNorm
    quantize(&xq, x, GS);
    matmul<dim, vocab_size>(out, xq.q, xq.s, w->wcls->q, w->wcls->s);  // Classifier
}
