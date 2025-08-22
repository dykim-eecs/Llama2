## run
g++ -std=c++17 host.cpp /home/dykim/let/build/k1/xcl2.cpp -o host \
-I/home/dykim/let/build/k1 \
-I/tools/Xilinx/Vitis_HLS/2022.2/include \
-I/opt/xilinx/xrt/include \
-L/opt/xilinx/xrt/lib \
-lOpenCL -lxrt_core -lxrt_coreutil -pthread
