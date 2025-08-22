
Vitis HLS Project README
This project demonstrates an optimized RMSNorm operation using Vitis HLS. The simulation commands have changed across different Vitis versions. Please refer to the instructions below to run the simulation correctly for your environment.

Simulation Instructions
Vitis HLS 2024.1 and Prior
For Vitis HLS versions prior to 2024.1, you can run the C-level simulation by directly calling the vitis_hls command with the TCL script.

vitis_hls -f ./csim.tcl

Vitis HLS 2024.1 and Later
Starting with version 2024.1, the recommended command for running HLS simulations is vitis-run, which is part of the unified Vitis tool flow.

vitis-run --mode hls --tcl ./csim.tcl

Testing Optimized Versions
This project allows you to switch between the original and optimized versions of the code using the OPTIMIZED environment variable.

To run the optimized version:
Set OPTIMIZED=1 before the vitis-run command.

OPTIMIZED=1 vitis-run --mode hls --tcl ./csim.tcl

To run the original version:
Set OPTIMIZED=0 before the vitis-run command.

OPTIMIZED=0 vitis-run --mode hls --tcl ./csim.tcl
