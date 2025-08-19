# ==== User Config ====
TARGET             ?= hw
PLATFORM           ?= xilinx_u250_gen3x16_xdma_4_1_202210_1
PLATFORM_REPO_PATH ?= /opt/xilinx/platforms
PFM                := $(PLATFORM_REPO_PATH)/$(PLATFORM)/$(PLATFORM).xpfm
KERNELS            := 1


# ==== Build rules via define + eval ====
define MAKE_KERNEL_RULE
build_k$(1):
	@echo ">> Building kernel $(1)..."
	@mkdir -p build/k$(1)

	v++ -c -k forward \
		-o build/k$(1)/forward.xo \
		--platform $(PFM) \
		--target $(TARGET) \
		--config config/multi_$(1).ini \
		-Ikernel \
		forward.cpp

	v++ -l \
		-o build/k$(1)/forward.xclbin \
		build/k$(1)/forward.xo \
		--platform $(PFM) \
		--target $(TARGET) \
		--config config/multi_$(1).ini
endef


# Generate rules dynamically
$(foreach K,$(KERNELS),$(eval $(call MAKE_KERNEL_RULE,$(K))))


.PHONY: all clean
all: $(foreach K,$(KERNELS),build_k$(K))

clean:
	rm -rf build *.log *.jou *.html *.xml *.json *~
