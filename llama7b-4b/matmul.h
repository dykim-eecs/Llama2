#include <cstring>
#include <cstdint>

// GS를 템플릿 인자로 받는 것이 더 좋지만, GS=64로 가정합니다.
template <int N, int D>
void matmul(float *xout, int8_t *xq, float *xs, int8_t *wq, float *ws) {
    constexpr int GS_const = 64;
    const int num_groups = N / GS_const;

    // 4비트 데이터를 담는 버퍼이므로 크기를 절반으로 변경
    int8_t localXQ[N / 2];
    float localXS[num_groups];
#pragma HLS ARRAY_PARTITION variable=localXQ cyclic factor=32 dim=1 // 8비트 정수에 2개씩 담기므로 factor를 절반으로
    
    // Load inputs into local buffers
    load_xq:
    for (int k = 0; k < N / 2; k++) { // 루프 횟수를 절반으로
        localXQ[k] = xq[k];
    }
    load_xs:
    for (int g = 0; g < num_groups; g++) {
        localXS[g] = xs[g];
    }

    // Main computation loop
    outer: for (int i = 0; i < D; i++) {
        // 4비트 데이터를 담는 버퍼이므로 크기를 절반으로 변경
        int8_t w_buffer[N / 2];
        float ws_buffer[num_groups];
#pragma HLS ARRAY_PARTITION variable=w_buffer cyclic factor=32 dim=1

        // Load the current row of the weight matrix and its scales
        load_wq:
        for (int k = 0; k < N / 2; k++) { // 루프 횟수를 절반으로
            w_buffer[k] = wq[i * (N / 2) + k]; // 인덱스도 N/2 기준으로 변경
        }
        load_ws:
        for (int g = 0; g < num_groups; g++) {
            ws_buffer[g] = ws[i * num_groups + g];
        }

        float val = 0.0f;
        groups: for (int g = 0; g < num_groups; g++) {
            int32_t ival = 0;
            matmul_inner: for (int k = 0; k < GS_const; k++) { // GS는 64로 고정
                #pragma HLS UNROLL
                int offset_packed = (g * GS_const + k) / 2; // 패킹된 버퍼의 인덱스
                
                // 4비트 값 두 개를 언패킹
                int8_t xq_unpacked, wq_unpacked;
                if ((g * GS_const + k) % 2 == 0) {
                    // 짝수 인덱스: 하위 4비트
                    xq_unpacked = localXQ[offset_packed] & 0x0F;
                    if (xq_unpacked & 0x08) xq_unpacked |= 0xF0;
                    
                    wq_unpacked = w_buffer[offset_packed] & 0x0F;
                    if (wq_unpacked & 0x08) wq_unpacked |= 0xF0;
                } else {
                    // 홀수 인덱스: 상위 4비트
                    xq_unpacked = (localXQ[offset_packed] >> 4) & 0x0F;
                    if (xq_unpacked & 0x08) xq_unpacked |= 0xF0;

                    wq_unpacked = (w_buffer[offset_packed] >> 4) & 0x0F;
                    if (wq_unpacked & 0x08) wq_unpacked |= 0xF0;
                }
                
                ival += static_cast<int32_t>(xq_unpacked) * static_cast<int32_t>(wq_unpacked);
            }
            val += static_cast<float>(ival) * localXS[g] * ws_buffer[g];
        }
        xout[i] = val;
    }
}