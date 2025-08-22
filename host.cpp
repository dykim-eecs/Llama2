//========================================================================================
// Vitis HLS Host Code
//
// This file demonstrates how to write a host code for a Vitis HLS kernel.
// It uses XRT (Xilinx Runtime) to manage the FPGA device, allocate and transfer
// memory buffers for the kernel, and execute the kernel.
//
// The code is based on the structure of the provided HLS kernel and testbench.
//========================================================================================
// Include necessary headers
#include "xcl2.hpp"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <numeric>
#include <cstring>
#include <cstdlib>
#include <cmath>
// Include structs and constants from the HLS kernel code.
// These headers are part of your Vitis HLS project.
#include "forward.h"
#include "config.h"
//
// Reusable functions and structs from the testbench file
//
// Note: The structs for QuantizedTensor, TransformerWeights, and Transformer
// are already defined in a header file included by forward.h (e.g., typedefs.h),
// so redefining them here would cause a redefinition error. They have been
// removed from this file.
// Structs and functions for the tokenizer
struct TokenIndex {
    char *str;
    int id;
};
struct Tokenizer {
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512];
};
int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}
void build_tokenizer(Tokenizer *t, const std::string& tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = (char **)malloc(vocab_size * sizeof(char *));
    t->vocab_scores = (float *)malloc(vocab_size * sizeof(float));
    t->sorted_vocab = NULL;
    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    FILE *file = fopen(tokenizer_path.c_str(), "rb");
    if (!file) {
        std::cerr << "Failed to open tokenizer: " << tokenizer_path << "\n";
        exit(1);
    }
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
char *decode(Tokenizer *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return (char *)"[INVALID]";
    return t->vocab[token];
}
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    TokenIndex tok = {.str = str};
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res ? res->id : -1;
}
void encode(Tokenizer *t, char *text, int8_t bos, int8_t eos, int *tokens, int *n_tokens) {
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
void softmax(float *x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) if (x[i] > max_val) max_val = x[i];
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) x[i] /= sum;
}
// Function to read model weights from a checkpoint file
template <int dim, int hidden_dim, int n_layers, int n_heads, int n_kv_heads, int vocab_size, int seq_len, int GS>
void read_checkpoint(const std::string& checkpoint, Config* config,
                     TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>* weights) {
    FILE* file = fopen(checkpoint.c_str(), "rb");
    if (!file) {
        std::cerr << "Error: Could not open file " << checkpoint << "\n";
        exit(1);
    }
    uint32_t magic;
    fread(&magic, sizeof(uint32_t), 1, file);
    if (magic != 0x616b3432) {
        std::cerr << "Error: Bad magic number.\n";
        exit(1);
    }
    int version;
    fread(&version, sizeof(int), 1, file);
    if (version != 2) {
        std::cerr << "Error: Unsupported version " << version << "\n";
        exit(1);
    }
    int header_size = 256;
    fread(config, sizeof(Config) - sizeof(int), 1, file);
    uint8_t shared_classifier;
    fread(&shared_classifier, sizeof(uint8_t), 1, file);
    int group_size;
    fread(&group_size, sizeof(int), 1, file);
    config->GS = GS;
    fseek(file, header_size, SEEK_SET);
    auto read_qtensor = [&](auto& tensor, int count, int size_each) {
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
// Explicit instantiation to ensure template code is generated for these parameters
template void read_checkpoint<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(
    const std::string&, Config*, TransformerWeights<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>*);
// === Main host code starts here ===
int main(int argc, char* argv[]) {
    // 1. Command-line argument parsing
    if (argc < 4) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File> <Model File> <Tokenizer File> [Prompt] [Steps]" << std::endl;
        return EXIT_FAILURE;
    }
    std::string xclbin_file(argv[1]);
    std::string model_file(argv[2]);
    std::string tokenizer_file(argv[3]);
    std::string prompt_str = (argc > 4) ? argv[4] : "I am happy";
    int steps = (argc > 5) ? std::stoi(argv[5]) : 10;
    const char *prompt = prompt_str.c_str();
    // 2. XRT device initialization
    std::cout << "Finding and initializing FPGA device." << std::endl;
    cl_int err;
    auto devices = xcl::get_xil_devices();
    if (devices.empty()) {
        std::cerr << "Error: No Xilinx devices found." << std::endl;
        return EXIT_FAILURE;
    }
    cl::Device device = devices[0];
    cl::Context context(device, nullptr, nullptr, nullptr, &err);
    cl::CommandQueue q(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, &err);
    cl::Program::Binaries bins = xcl::import_binary_file(xclbin_file);
    cl::Program program(context, {device}, bins, nullptr, &err);
   
    cl::Kernel krnl_forward(program, "forward", &err);
    std::cout << "Kernel loaded." << std::endl;
    // 3. Allocate host memory and load model/tokenizer data
    std::cout << "Allocating host memory and loading model data." << std::endl;
   
    // Allocate aligned memory using posix_memalign to avoid unaligned pointer warnings
    // Use unique_ptr for automatic memory management with a custom free deleter.
    void *raw_transformer_ptr;
    if (posix_memalign(&raw_transformer_ptr, 4096, sizeof(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>)) != 0) {
        std::cerr << "Error: posix_memalign failed for transformer." << std::endl;
        return EXIT_FAILURE;
    }
    std::unique_ptr<Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>, decltype(&free)>
        transformer(reinterpret_cast<Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>*>(raw_transformer_ptr), free);
   
    // Fix: Pass the address of the config member, not the entire transformer object.
    read_checkpoint<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>(model_file, &transformer->config, &transformer->weights);
    // Initialize tokenizer and encode prompt
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_file, vocab_size);
    std::vector<int> prompt_tokens(seq_len);
    int num_prompt_tokens = 0;
    encode(&tokenizer, (char*)prompt, 1, 0, prompt_tokens.data(), &num_prompt_tokens);
    if (num_prompt_tokens <= 0) {
        std::cerr << "Error: Failed to encode prompt." << std::endl;
        return EXIT_FAILURE;
    }
   
    // Memory for key_cache, value_cache, and out
    size_t kv_cache_size = n_layers * seq_len * ((dim * n_kv_heads) / n_heads);
    size_t out_size = vocab_size;
    // Use posix_memalign for aligned memory
    void *raw_key_cache, *raw_value_cache, *raw_out;
    if (posix_memalign(&raw_key_cache, 4096, kv_cache_size * sizeof(float)) != 0 ||
        posix_memalign(&raw_value_cache, 4096, kv_cache_size * sizeof(float)) != 0 ||
        posix_memalign(&raw_out, 4096, out_size * sizeof(float)) != 0) {
        std::cerr << "Error: posix_memalign failed for caches." << std::endl;
        return EXIT_FAILURE;
    }
    std::unique_ptr<float, decltype(&free)> key_cache(reinterpret_cast<float*>(raw_key_cache), free);
    std::unique_ptr<float, decltype(&free)> value_cache(reinterpret_cast<float*>(raw_value_cache), free);
    std::unique_ptr<float, decltype(&free)> out(reinterpret_cast<float*>(raw_out), free);
   
    std::memset(key_cache.get(), 0, kv_cache_size * sizeof(float));
    std::memset(value_cache.get(), 0, kv_cache_size * sizeof(float));
    std::memset(out.get(), 0, out_size * sizeof(float));
    // 4. Allocate device memory (buffers) and link with host buffers
    std::cout << "Allocating device memory buffers." << std::endl;
   
    cl::Buffer transformer_buf(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
                               sizeof(Transformer<dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len, GS>), transformer.get(), &err);
    cl::Buffer key_cache_buf(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                             kv_cache_size * sizeof(float), key_cache.get(), &err);
    cl::Buffer value_cache_buf(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR,
                               kv_cache_size * sizeof(float), value_cache.get(), &err);
    cl::Buffer out_buf(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
                       out_size * sizeof(float), out.get(), &err);
    // 5. Set kernel arguments
    krnl_forward.setArg(0, transformer_buf);
    krnl_forward.setArg(3, key_cache_buf);
    krnl_forward.setArg(4, value_cache_buf);
    krnl_forward.setArg(5, out_buf);
    // 6. Kernel execution loop
    std::cout << "\nStarting kernel execution." << std::endl;
   
    int token = prompt_tokens[0];
   
    std::cout << "Input: " << prompt << std::endl;
    std::cout << "Output: " << std::endl;
    double total_kernel_time_ms = 0.0; // 총 kernel 시간 (ms)
    int generated_tokens = 0; // 생성된 토큰 수
    for (int pos = 0; pos < steps; pos++) {
        // Correctly set dynamic arguments (token is arg 1, pos is arg 2)
        krnl_forward.setArg(1, token);
        krnl_forward.setArg(2, pos);
        // Transfer buffers to device
        q.enqueueMigrateMemObjects({transformer_buf, key_cache_buf, value_cache_buf}, 0);
        q.finish();
       
        // Execute the kernel with profiling
        cl::Event kernel_event;
        q.enqueueTask(krnl_forward, NULL, &kernel_event);
        kernel_event.wait(); // 대기
        // 시간 계산 (ns -> ms)
        cl_ulong start_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
        cl_ulong end_time = kernel_event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
        double kernel_time_ms = (end_time - start_time) / 1e6; // ns to ms
        total_kernel_time_ms += kernel_time_ms;
       
        // Transfer results back to host
        q.enqueueMigrateMemObjects({key_cache_buf, value_cache_buf, out_buf}, CL_MIGRATE_MEM_OBJECT_HOST);
        q.finish();
        // Process results (softmax and select next token)
        softmax(out.get(), vocab_size);
        // Determine the next token
        int next;
        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            float max_val = out.get()[0];
            int max_idx = 0;
            for (int j = 1; j < out_size; j++) {
                if (out.get()[j] > max_val) {
                    max_val = out.get()[j];
                    max_idx = j;
                }
            }
            next = max_idx;
            generated_tokens++; // 생성 토큰 카운트
        }
        // Decode and print the token
        char *piece = decode(&tokenizer, token, next);
        if (next == 2) { // 2 is the EOS (End-of-Sentence) token.
            break;
        }
        std::cout << piece << std::endl;
        token = next;
    }
    // Token/s 계산 및 출력
    if (generated_tokens > 0) {
        double total_time_sec = total_kernel_time_ms / 1000.0;
        double tokens_per_sec = generated_tokens / total_time_sec;
        std::cout << "\nTokens per second: " << tokens_per_sec << std::endl;
    } else {
        std::cout << "\nNo tokens generated." << std::endl;
    }
    std::cout << "\nKernel execution complete." << std::endl;
   
    // std::unique_ptr handles memory deallocation automatically
    return EXIT_SUCCESS;
}