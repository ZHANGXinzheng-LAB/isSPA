#include "image.hpp"

Image::Image(const LST::Entry & input) : rpath(input.rpath), p(Params{input.defocus, input.dfdiff, input.dfang}), unused(input.unused) 
{
    auto image = std::make_unique<emdata>();
    // read from mrc file
    image->readImage(rpath.c_str(), input.unused);

    auto ptr = image->getData();
    auto width = p.width = image->header.nx;
    auto height = p.height = image->header.ny;
    std::printf("\nImage: %s, width: %zu, height: %zu\n", rpath.c_str(), width, height);

    data = std::make_unique<float[]>(width * height);
    std::memcpy(data.get(), ptr, sizeof(float) * width * height);
}