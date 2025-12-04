#pragma once

#include <cufft.h>

cufftHandle MakeFFTPlan(int dim0, int dim1, unsigned int size);