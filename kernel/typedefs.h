#include <stdint.h>
#include <stdio.h>

#ifndef TYPEDEFS
#define TYPEDEFS

struct Config {  // Model configuration
    int dim;          // Transformer dimension
    int hidden_dim;   // For FFN layers
    int n_layers;     // Number of layers
    int n_heads;      // Number of query heads
    int n_kv_heads;   // Number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;   // Vocabulary size, usually 256 (byte-level)
    int seq_len;      // Max sequence length
    int GS;           // Group size
};

template <int SIZE>
struct QuantizedTensor {  // Quantized tensor structure
    int8_t q[SIZE];       // Quantized values
    float s[SIZE];        // Scaling factors
};

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct TransformerWeights {  // Transformer model weights
    // Token embedding table
    QuantizedTensor<vocab_size * dim> q_tokens[1];  // (vocab_size, dim)
    float token_embedding_table[vocab_size * dim];  // Same, but dequantized
    // Weights for RMSNorms
    float rms_att_weight[n_layers * dim];  // (layer, dim) RMSNorm weights
    float rms_ffn_weight[n_layers * dim];  // (layer, dim)
    // Weights for matmuls. Note dim == n_heads * head_size
    QuantizedTensor<dim * (dim / n_heads) * n_heads> wq[n_layers];    // (layer, dim, n_heads * head_size)
    QuantizedTensor<dim * (dim / n_heads) * n_kv_heads> wk[n_layers]; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor<dim * (dim / n_heads) * n_kv_heads> wv[n_layers]; // (layer, dim, n_kv_heads * head_size)
    QuantizedTensor<n_heads * dim * (dim / n_heads)> wo[n_layers];     // (layer, dim, n_heads * head_size)
    // Weights for FFN
    QuantizedTensor<dim * hidden_dim> w1[n_layers];  // (layer, hidden_dim, dim)
    QuantizedTensor<dim * hidden_dim> w2[n_layers];  // (layer, dim, hidden_dim)
    QuantizedTensor<dim * hidden_dim> w3[n_layers];  // (layer, hidden_dim, dim)
    // Final RMSNorm
    float rms_final_weight[dim];  // (dim,)
    // (Optional) classifier weights for the logits, on the last layer
    QuantizedTensor<vocab_size * dim> wcls[1];
};

// ----------------------------------------------------------------------------
// Transformer model
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
struct Transformer {  // Transformer model structure
    Config config;    // Hyperparameters of the architecture
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> weights;  // Model weights
    // Some more state needed to properly clean up the memory mapping (sigh)
    // int fd; // File descriptor for memory mapping
    // float *data; // Memory mapped data pointer
    // ssize_t file_size; // Size of the checkpoint file in bytes
};

#endif