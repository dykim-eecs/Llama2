# run_hls.tcl
open_project proj_forward
set_top forward
add_files forward.cpp
add_files -tb testbench.cpp

open_solution "sol1"
set_part {xcu250-figd2104-2L-e}
create_clock -period 3.3 -name default

csim_design    ;# C simulation
#csynth_design  ;# HLS synthesis
#cosim_design   ;# C/RTL co-simulation
#export_design -format xo
exit
