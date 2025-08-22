#include <iostream>
#include <cmath>
#include <numeric>

#define S_SIZE 768

extern "C" void rmsnorm_wrapper(float o_out[768], float x_in[768], float weight_in[768]);

// A reference C++ implementation to verify the HLS output.
// This function performs the same calculation as the HLS kernel but in a simple C++ way.
void rmsnorm_ref(float o[S_SIZE], float x[S_SIZE], float weight[S_SIZE]) {
    // 1. Calculate sum of squares
    float ss = 0.0f;
    for (int i = 0; i < S_SIZE; i++) {
        ss += x[i] * x[i];
    }
    
    // 2. Normalize and scale
    float ss_scaled = 1.0f / sqrtf(ss / S_SIZE + 1e-5f);
    
    for (int i = 0; i < S_SIZE; i++) {
        o[i] = weight[i] * (ss_scaled * x[i]);
    }
}


int main() {
    // Arrays for input, HLS output, and reference output
    float x[S_SIZE];
    float weight[S_SIZE];
    float o_hls[S_SIZE];
    float o_ref[S_SIZE];
    
    // Initialize input data
    std::cout << "Initializing input data..." << std::endl;
    for (int i = 0; i < S_SIZE; i++) {
        x[i] = (float)i / 100.0f;
        weight[i] = (float)(S_SIZE - i) / 100.0f;
    }

    // Call the HLS-optimized function via the wrapper
    std::cout << "Executing HLS kernel..." << std::endl;
    rmsnorm_wrapper(o_hls, x, weight);

    // Call the reference function
    std::cout << "Executing reference model..." << std::endl;
    rmsnorm_ref(o_ref, x, weight);

    // Verification step: Compare HLS output with the reference output
    std::cout << "Comparing HLS output with reference output..." << std::endl;
    int errors = 0;
    const float THRESHOLD = 1e-6; // Tolerance for floating-point comparison
    for (int i = 0; i < S_SIZE; i++) {
        float diff = std::abs(o_hls[i] - o_ref[i]);
        if (diff > THRESHOLD) {
            errors++;
            std::cout << "ERROR: Mismatch at index " << i << ". HLS: " << o_hls[i] << ", Ref: " << o_ref[i] << std::endl;
        }
    }

    if (errors == 0) {
        std::cout << "INFO: Test passed! The HLS kernel output matches the reference model." << std::endl;
        return 0; // Success
    } else {
        std::cout << "ERROR: Test failed with " << errors << " mismatches." << std::endl;
        return 1; // Failure
    }
}
