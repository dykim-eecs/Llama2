
## 가중치 링크

https://drive.google.com/drive/folders/1L1EirYH7ygG8Vnf5wjBya-lNNzCfKDmX

## Simulation & Testing Instructions


This project provides an example for running C-level simulation in **Vitis HLS**. The command to run simulation depends on the version of Vitis being used.

### **Before 2025 (Vitis HLS 2024.1 and earlier)**

Use the `vitis_hls` command:

```bash
vitis_hls -f ./csim.tcl
```

---

### **After 2025 (Vitis HLS 2025.1 and later)**

Starting from **Vitis 2025.1**, the unified Vitis tool flow is recommended. Use the `vitis-run` command instead:

```bash
vitis-run --mode hls --tcl ./csim.tcl
```

</details>
