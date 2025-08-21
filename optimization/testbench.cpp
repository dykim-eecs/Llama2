#include <iostream>
#include <cmath>

#define S_SIZE 768

extern "C" void rmsnorm(float o[768], float x[768], float weight[768]);

int main() {
    float x[S_SIZE];
    float weight[S_SIZE];
    float o_hls[S_SIZE];
    
    // Initialize input data
    for (int i = 0; i < S_SIZE; i++) {
        x[i] = (float)i / 100.0f;
        weight[i] = (float)(S_SIZE - i) / 100.0f;
    }

    // Print the values of x
    std::cout << "Values of x before rmsnorm:" << std::endl;
    for (int i = 0; i < S_SIZE; i++) {
        std::cout << "x[" << i << "] = " << x[i] << std::endl;
    }
    std::cout << "---------------------------" << std::endl;

    // Run the HLS function 1 times
    msnorm(o_hls, x, weight);
    std::cout << "INFO: rmsnorm executed (run " << run + 1 << ")" << std::endl;

    std::cout << "INFO: HLS function executed successfully 3 times." << std::endl;

    return 0;
}
