#pragma once

#include "typedefs.h"

static constexpr int dim = 768;          // Model embedding dimension
static constexpr int hidden_dim = 2048;  // Hidden dimension for feed-forward layers
static constexpr int n_layers = 12;      // Number of transformer layers
static constexpr int n_heads = 12;       // Number of attention heads
static constexpr int n_kv_heads = 12;    // Number of key-value attention heads
static constexpr int vocab_size = 32000; // Vocabulary size
static constexpr int seq_len = 1024;     // Maximum sequence length
static constexpr int GS = 64;            // Group size (e.g., for quantization or parallelism)

constexpr Config config = {
    .dim = dim,
    .hidden_dim = hidden_dim,
    .n_layers = n_layers,
    .n_heads = n_heads,
    .n_kv_heads = n_kv_heads,
    .vocab_size = vocab_size,
    .seq_len = seq_len,
    .GS = GS,
};