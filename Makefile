# ==== User Config ====
TARGET             ?= hw
PLATFORM           ?= kv260_platform
PLATFORM_REPO_PATH ?= platform
PFM                := $(PLATFORM_REPO_PATH)/$(PLATFORM)/$(PLATFORM).xpfm
KERNELS            := 1

# kv260 : xck26-sfvc784-2LV-C

# ==== Build rules via define + eval ====
define MAKE_KERNEL_RULE
build_k$(1):
	@echo ">> Building kernel $(1)..."
	@mkdir -p build/k$(1)

	v++ -c -k forward \
		-o build/k$(1)/forward.xo \
		--platform $(PFM) \
		--target $(TARGET) \
		-Ikernel \
		kernel/forward.cpp

	v++ -l \
		-o build/k$(1)/forward.xclbin \
		build/k$(1)/forward.xo \
		--platform $(PFM) \
		--target $(TARGET)
endef


# Generate rules dynamically
$(foreach K,$(KERNELS),$(eval $(call MAKE_KERNEL_RULE,$(K))))


.PHONY: all clean
all: $(foreach K,$(KERNELS),build_k$(K))

clean:
	rm -rf build *.log *.jou *.html *.xml *.json *~ *.Xil *.ipcache *_x





