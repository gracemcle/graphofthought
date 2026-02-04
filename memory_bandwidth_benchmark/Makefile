# Memory Bandwidth Benchmark Suite Makefile
# Adjust SM_ARCH for your GPU

# Auto-detect GPU architecture (fallback to sm_70)
SM_ARCH ?= sm_80

# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -O3 -arch=$(SM_ARCH) -lineinfo
NVCC_FLAGS += -Xcompiler -Wall

# For newer GPUs, enable more optimizations
ifeq ($(SM_ARCH),sm_80)
    NVCC_FLAGS += -Xptxas=-v
endif
ifeq ($(SM_ARCH),sm_86)
    NVCC_FLAGS += -Xptxas=-v
endif
ifeq ($(SM_ARCH),sm_89)
    NVCC_FLAGS += -Xptxas=-v
endif
ifeq ($(SM_ARCH),sm_90)
    NVCC_FLAGS += -Xptxas=-v
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin

# Source files
SOURCES = $(SRC_DIR)/bandwidth_kernels.cu
EXECUTABLE = $(BIN_DIR)/bandwidth_benchmark

# Default target
all: directories $(EXECUTABLE)

directories:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR)

$(EXECUTABLE): $(SOURCES)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Run with default settings
run: $(EXECUTABLE)
	./$(EXECUTABLE)

# Run with larger array (512 MB)
run-large: $(EXECUTABLE)
	./$(EXECUTABLE) 512

# Run with smaller array (fits in L2)
run-small: $(EXECUTABLE)
	./$(EXECUTABLE) 32

# Profile with nsight compute (requires ncu)
profile: $(EXECUTABLE)
	ncu --set full -o bandwidth_profile ./$(EXECUTABLE) 64

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)

# Show GPU info
gpu-info:
	nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

# Help
help:
	@echo "Memory Bandwidth Benchmark Suite"
	@echo ""
	@echo "Targets:"
	@echo "  all          Build the benchmark (default)"
	@echo "  run          Run with default settings (256 MB arrays)"
	@echo "  run-large    Run with 512 MB arrays"
	@echo "  run-small    Run with 32 MB arrays (may fit in L2)"
	@echo "  profile      Profile with NVIDIA Nsight Compute"
	@echo "  clean        Remove build artifacts"
	@echo "  gpu-info     Show GPU information"
	@echo ""
	@echo "Variables:"
	@echo "  SM_ARCH      GPU architecture (default: sm_80)"
	@echo ""
	@echo "Example:"
	@echo "  make SM_ARCH=sm_86 run"

.PHONY: all directories run run-large run-small profile clean gpu-info help
