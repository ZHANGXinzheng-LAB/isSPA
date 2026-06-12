SHELL=/usr/bin/bash

WD := $(shell pwd)
TARGET_EXEC := main

LIB_HDF5 := $(WD)/../hdf5/lib
INCLUDE_HDF5 := $(WD)/../hdf5/include

LDFLAGS := -lcufft -L$(LIB_HDF5) -lhdf5 -lgomp
CXXFLAGS := -std=c++17 -O3 -Xcompiler -fopenmp -I$(INCLUDE_HDF5) -Iinclude -Iinclude/EMReader -Iinclude/utils -arch=sm_86
CXX := nvcc

BUILD_DIR := $(WD)/build
SRCS := \
		src/fft.cu \
		src/helper.cu \
		src/kernels.cu \
		src/norm.cu \
		src/EMReader/DataReader2.cpp \
		src/EMReader/emdata.cpp \
		src/EMReader/emhdf.cpp \
		src/EMReader/emhdf2.cpp \
		src/utils/image.cpp \
		src/utils/templates.cpp \
		src/utils/tileimages.cpp \
		src/utils/utils.cpp

MAIN := src/main.cu
PROJECT3D := test/project3d.cpp

OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJS) $(MAIN) -o $@ $(LDFLAGS)

$(BUILD_DIR)/%.cu.o: %.cu
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/%.cpp.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

project3d: $(OBJS)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(OBJS) $(PROJECT3D) -o project3d $(LDFLAGS)

install: $(BUILD_DIR)/$(TARGET_EXEC)
	ln $(BUILD_DIR)/$(TARGET_EXEC) $(BUILD_DIR)/isSPA

test: install
	$(WD)/test.sh

.PHONY: clean
clean:
	-rm -r $(BUILD_DIR)
