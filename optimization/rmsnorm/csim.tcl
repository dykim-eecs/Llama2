# ==============================
# csim.tcl
# ==============================

# 환경변수 읽기
if {[info exists ::env(OPTIMIZED)] && $::env(OPTIMIZED)} {
    set optimized true
} else {
    set optimized false
}
puts ">> csim.tcl: optimized = $optimized"

# 프로젝트 열기
open_project proj_rmsnorm

if { $optimized } {

    puts ">> Using OPTIMIZED rmsnorm"
    set_top rmsnorm_wrapper
    add_files optimized_rmsnorm.cpp
    add_files -tb testbench.cpp -cflags "-DUSE_OPTIMIZED"

    open_solution "sol_optimized"
    set_part {xcu250-figd2104-2L-e}
    create_clock -period 3.3 -name default

} else {

    puts ">> Using ORIGINAL rmsnorm"
    set_top rmsnorm_wrapper
    add_files original_rmsnorm.cpp
    add_files -tb testbench.cpp

    open_solution "sol_original"
    set_part {xcu250-figd2104-2L-e}
    create_clock -period 3.3 -name default
}

# 실행 단계
csim_design
#csynth_design
#cosim_design
#export_design -format xo

exit

