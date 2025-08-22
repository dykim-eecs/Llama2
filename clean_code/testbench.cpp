#include "forward.h"
#include "config.h"
#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <cstdlib>
#include <fcntl.h>
#include <unistd.h>

struct TokenIndex {  // Token index structure
    char *str;
    int id;
};

struct Tokenizer {  // Tokenizer structure
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};

int compare_tokens(const void *a, const void *b) {  // Comparison function for tokens
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, std::string tokenizer_path, int vocab_size) {  // Build tokenizer from file
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) { std::cerr << "Failed to open tokenizer.\n"; exit(1); }
    fread(&t->max_token_length, sizeof(int), 1, file);
    int len;
    for (int i = 0; i < vocab_size; i++) {
        fread(t->vocab_scores + i, sizeof(float), 1, file);
        fread(&len, sizeof(int), 1, file);
        t->vocab[i] = (char *)malloc(len + 1);
        fread(t->vocab[i], len, 1, file);
        t->vocab[i][len] = '\0';
    }
    fclose(file);
}

char *decode(Tokenizer *t, int prev_token, int token) {  // Decode token to string
    if (token < 0 || token >= t->vocab_size) return (char *)"[INVALID]";
    return t->vocab[token];
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {  // Lookup string in sorted vocab
    TokenIndex tok = {.str = str};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}

void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {  // Encode text to tokens
    if (text == NULL) { std::cerr << "NULL text\n"; exit(1); }
    if (t->sorted_vocab == NULL) {
        t->sorted_vocab = (TokenIndex *)malloc(t->vocab_size * sizeof(TokenIndex));
        for (int i = 0; i < t->vocab_size; i++) {
            t->sorted_vocab[i].str = t->vocab[i];
            t->sorted_vocab[i].id = i;
        }
        qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
    }
    char *str_buffer = (char *)malloc((t->max_token_length * 2 + 3));
    size_t str_len = 0;
    *n_tokens = 0;
    if (bos) tokens[(*n_tokens)++] = 1;
    if (text[0] != '\0') {
        int dummy_prefix = str_lookup((char *)" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }
    for (char *c = text; *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) str_len = 0;
        str_buffer[str_len++] = *c;
        str_buffer[str_len] = '\0';
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4) continue;
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
        if (id != -1) tokens[(*n_tokens)++] = id;
        else {
            for (int i = 0; i < str_len; i++)
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
        }
        str_len = 0;
    }
    while (true) {
        float best_score = -1e10;
        int best_id = -1, best_idx = -1;
        for (int i = 0; i < (*n_tokens - 1); i++) {
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }
        if (best_idx == -1) break;
        tokens[best_idx] = best_id;
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++) tokens[i] = tokens[i + 1];
        (*n_tokens)--;
    }
    if (eos) tokens[(*n_tokens)++] = 2;
    free(str_buffer);
}

void softmax(float *x, int size) {  // Softmax function
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}

// === Forward declaration for build_transformer ===
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string path);

// === Main function ===
int main() {
    auto *transformer = new Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>();
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, "tokenizer.bin", vocab_size);
    build_transformer(transformer, "model.bin");
    const char *prompt = (char *)"I am happy";
    int *prompt_tokens = (int *)malloc(seq_len * sizeof(int));
    int num_prompt_tokens = 0;
    encode(&tokenizer, (char *)prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
    float *key_cache = (float *)calloc(n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *value_cache = (float *)calloc(n_layers * seq_len * ((dim * n_kv_heads) / n_heads), sizeof(float));
    float *logits = (float *)calloc(vocab_size, sizeof(float));
    int token = prompt_tokens[0];
    int pos = 0;
    int next = 0;
    int steps = 10;
    std::cout << "Input: " << prompt << std::endl;
    std::cout << "Output: " << std::endl;
    for (int i = 0; i < steps; i++) {
        forward(transformer, token, pos, key_cache, value_cache, logits);
        softmax(logits, vocab_size);
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            float max_val = logits[0];
            int max_idx = 0;
            for (int j = 1; j < vocab_size; j++) {
                if (logits[j] > max_val) {
                    max_val = logits[j];
                    max_idx = j;
                }
            }
            next = max_idx;
        }
        char *piece = decode(&tokenizer, token, next);
        if (next == 2) break;
        std::cout << piece << std::endl;
        token = next;
        pos++;
    }
    std::cout << std::endl;
    delete transformer;
    free(prompt_tokens);
    free(key_cache);
    free(value_cache);
    free(logits);
    return 0;
}

// === Template function definitions must follow ===
// Explicit instantiation to ensure template code is generated for these parameters
template void build_transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(
    Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>*, std::string);

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(std::string checkpoint, Config *config,
    TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *weights) {  // Read model checkpoint
    FILE *file = fopen(checkpoint.c_str(), "rb");
    if (!file) { fprintf(stderr, "Couldn't open %s\n", checkpoint.c_str()); exit(1); }
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(1); }
    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 2) { fprintf(stderr, "Unsupported version %d\n", version); exit(1); }
    int header_size = 256;
    fread(config, sizeof(Config) - sizeof(int), 1, file);
    uint8_t shared_classifier;
    fread(&shared_classifier, sizeof(uint8_t), 1, file);
    int group_size;
    fread(&group_size, sizeof(int), 1, file);
    config->GS = GS;
    fseek(file, header_size, SEEK_SET);
    auto read_qtensor = [&](auto &tensor, int count, int size_each) {
        for (int i = 0; i < count; i++) {
            fread(tensor[i].q, sizeof(int8_t), size_each, file);
            fread(tensor[i].s, sizeof(float), size_each / GS, file);
        }
    };
    fread(weights->rms_att_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_ffn_weight, sizeof(float), n_layers * dim, file);
    fread(weights->rms_final_weight, sizeof(float), dim, file);
    read_qtensor(weights->q_tokens, 1, vocab_size * dim);
    dequantize<vocab_size * dim>(weights->q_tokens, weights->token_embedding_table, GS);
    read_qtensor(weights->wq, n_layers, dim * dim);
    read_qtensor(weights->wk, n_layers, dim * dim);
    read_qtensor(weights->wv, n_layers, dim * dim);
    read_qtensor(weights->wo, n_layers, dim * dim);
    read_qtensor(weights->w1, n_layers, dim * hidden_dim);
    read_qtensor(weights->w2, n_layers, hidden_dim * dim);
    read_qtensor(weights->w3, n_layers, dim * hidden_dim);
    if (shared_classifier) {
        std::memcpy(weights->wcls, weights->q_tokens, sizeof(QuantizedTensor<vocab_size * dim>));
    } else {
        read_qtensor(weights->wcls, 1, dim * vocab_size);
    }
    fclose(file);
}

template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void build_transformer(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS> *t, std::string path) {  // Build transformer from checkpoint
    read_checkpoint<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(path, &t->config, &t->weights);
}