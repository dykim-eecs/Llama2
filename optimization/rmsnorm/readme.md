## Simulation & Testing Instructions

<details>
<summary>Click to expand</summary>

이 프로젝트는 **Vitis HLS** 환경에서 RMSNorm 연산을 최적화한 버전을 시뮬레이션하는 예제를 제공합니다. Vitis 버전에 따라 시뮬레이션 명령어가 다르므로 아래 지침을 참고하세요.

### Vitis HLS 2024.1 and Prior
2024.1 이전 버전에서는 `vitis_hls` 명령어를 사용하여 C-level 시뮬레이션을 실행할 수 있습니다:
```bash
vitis_hls -f ./csim.tcl
Vitis HLS 2024.1 and Later
2024.1 버전부터는 통합된 Vitis 툴 플로우에 포함된 vitis-run 명령어 사용이 권장됩니다:

bash
복사
편집
vitis-run --mode hls --tcl ./csim.tcl
Testing Optimized Versions
이 프로젝트는 OPTIMIZED 환경 변수를 통해 원본 코드와 최적화된 코드를 전환하며 테스트할 수 있습니다.

Run Optimized Version
최적화된 버전을 실행하려면 OPTIMIZED=1을 설정하세요:

bash
복사
편집
OPTIMIZED=1 vitis-run --mode hls --tcl ./csim.tcl
Run Original Version
원본 버전을 실행하려면 OPTIMIZED=0을 설정하세요:

bash
복사
편집
OPTIMIZED=0 vitis-run --mode hls --tcl ./csim.tcl
</details> ```
