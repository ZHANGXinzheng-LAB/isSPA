SHELL=/usr/bin/bash

WD := $(shell pwd)
TARGET_EXEC := main
INSTALL_EXEC := scripts/isSPA

LDFLAGS := -lcufft
CXXFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp -Iinclude -Iinclude/EMReader -Iinclude/utils -arch=sm_86
CXX := nvcc

BUILD_DIR := $(WD)/build
SRCS := \
		src/fft.cu \
		src/helper.cu \
		src/kernels.cu \
		src/norm.cu \
		src/EMReader/DataReader2.cpp \
		src/EMReader/emdata.cpp \
		src/utils/image.cpp \
		src/utils/templates.cpp \
		src/utils/utils.cpp \
		src/main.cu

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJS) $(MAIN) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

install: $(BUILD_DIR)/$(TARGET_EXEC) 
	ln -fsr $(BUILD_DIR)/$(TARGET_EXEC) $(INSTALL_EXEC)

.PHONY: clean
clean:
	-rm -r $(BUILD_DIR) $(INSTALL_EXEC)
