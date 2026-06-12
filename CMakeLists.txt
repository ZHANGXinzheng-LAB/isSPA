cmake_minimum_required(VERSION 3.10)

project(GisSPA LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH /usr/local/musa/cmake)

find_package(MUSA REQUIRED)

set(SRCS

  src/fft.cu

  src/helper.cu

  src/kernels.cu

  src/nonorm.cu

  src/norm.cu

  src/EMReader/DataReader2.cpp

  src/EMReader/emdata.cpp

  src/EMReader/emhdf.cpp

  src/EMReader/emhdf2.cpp

  src/utils/image.cpp

  src/utils/templates.cpp

  src/utils/tileimages.cpp

  src/utils/utils.cpp

  src/main.cu
)

set_source_files_properties(${SRCS}
      PROPERTIES
      MUSA_SOURCE_PROPERTY_FORMAT OBJ

  )
musa_include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include/EMReader ${CMAKE_CURRENT_SOURCE_DIR}/include/utils /usr/lib/gcc/x86_64-linux-gnu/9/include ${CMAKE_CURRENT_SOURCE_DIR}/../hdf5/include)

set(MUSA_MCC_FLAGS --offload-arch=mp_22 -fopenmp -std=c++17)

musa_add_executable(main ${SRCS})

target_link_libraries(main

  pthread

  mufft

  musart

  gomp

  hdf5
  iomp5
)

target_link_directories(main PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../hdf5/lib
        /usr/local/musa/lib
        )

