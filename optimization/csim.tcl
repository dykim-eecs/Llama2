# run_hls.tcl
# Usage: vivado_hls -f run_hls.tcl <kernel_file_name> <top_function_name>
# Example: vivado_hls -f run_hls.tcl original_rmsnorm.cpp rmsnorm

# Check if two arguments are provided
if { $argc != 2 } {
    puts "Error: Invalid number of arguments."
    puts "Usage: vivado_hls -f run_hls.tcl <kernel_file_name> <top_function_name>"
    puts "Example: vivado_hls -f run_hls.tcl original_rmsnorm.cpp rmsnorm"
    exit
}

# Get the arguments
set kernel_file_name [lindex $argv 0]
set top_function_name [lindex $argv 1]

open_project proj_rmsnorm

# Add the specified kernel file
add_files $kernel_file_name
set_top $top_function_name

# Add testbench file
add_files -tb testbench.cpp

open_solution "sol1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.3 -name default

csim_design    ;# C simulation
#csynth_design  ;# HLS synthesis
#cosim_design   ;# C/RTL co-simulation
#export_design -format xo
exit
