usage ./csim.tcl <testbench_name> <top_function_name>


vitis_hls -f ./csim.tcl original_rmsnorm.cpp rmsnorm


vitis_hls -f ./csim.tcl optimized_rmsnorm.cpp rmsnorm


vitis-run --mode hls --tcl ./csim.tcl original_rmsnorm.cpp rmsnorm


vitis-run --mode hls --tcl ./csim.tcl optimized_rmsnorm.cpp rmsnorm
