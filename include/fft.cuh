#pragma once

#include <cufft.h>

cufftHandle MakeFFTPlan(int dim0, int dim1, int size);