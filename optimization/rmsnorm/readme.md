## Simulation & Testing Instructions


This project demonstrates an optimized RMSNorm operation using **Vitis HLS**. The simulation commands differ depending on the Vitis version. Please follow the instructions below according to your environment.

### Vitis HLS 2024.1 and Prior

For Vitis HLS versions prior to 2024.1, you can run the C-level simulation using the `vitis_hls` command:

```bash
vitis_hls -f ./csim.tcl
```

### Vitis HLS 2024.1 and Later

Starting with version 2024.1, the recommended command for running HLS simulations is `vitis-run`, which is part of the unified Vitis tool flow:

```bash
vitis-run --mode hls --tcl ./csim.tcl
```

### Testing Optimized Versions

This project allows you to switch between the original and optimized versions of the code using the **OPTIMIZED** environment variable.

* Run Optimized Version
  To run the optimized version, set `OPTIMIZED=1`:

  ```bash
  OPTIMIZED=1 vitis-run --mode hls --tcl ./csim.tcl
  ```

* Run Original Version
  To run the original version, set `OPTIMIZED=0`:

  ```bash
  OPTIMIZED=0 vitis-run --mode hls --tcl ./csim.tcl
  ```

</details>
