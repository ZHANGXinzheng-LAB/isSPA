#include "tileimages.hpp"

#include "emdata.h"

TileImages::TileImages(const LST::Entry & input)
    : rpath(input.rpath),
      p(Params{input.defocus, input.dfdiff, input.dfang}),
      unused(input.unused) 
{
  auto image = std::make_unique<emdata>();
  // read from mrc file
  image->readImage(rpath.c_str(), input.unused);

  auto ptr = image->getData();
  auto width = p.width = image->header.nx;
  auto height = p.height = image->header.ny;
  std::printf("Image: %s, width: %d, height: %d, ", rpath.c_str(), width, height);

  int cols = std::ceil(static_cast<double>(width) / tile_size);
  int rows = std::ceil(static_cast<double>(height) / tile_size);
  std::printf("split cols: %d, rows: %d, ", cols, rows);

  auto x_step = width / cols;
  auto y_step = height / rows;
  std::printf("x_step: %d, y_step: %d\n", x_step, y_step);

  int idx = 1;

  for (int j = 0; j < cols; ++j) 
  {
    int center_x{};
    int center_y{};

    // 计算每块照片的中心坐标
    auto x = j * tile_size;
    if (j == 0) 
    {
      center_x = x + tile_size / 2;
    } 
    else if (j == cols - 1) 
    {
      center_x = width - tile_size / 2;
    } 
    else 
    {
      center_x = x_step * (2 * j + 1) / 2;
    }

    for (int i = 0; i < rows; ++i) 
    {
      auto y = i * tile_size;
      if (i == 0) 
      {
        center_y = y + tile_size / 2;
      } 
      else if (i == rows - 1) 
      {
        center_y = height - tile_size / 2;
      } 
      else 
      {
        center_y = y_step * (2 * i + 1) / 2;
      }

      auto upper_left_x = std::max(center_x - tile_size / 2, 0);
      auto upper_left_y = std::max(center_y - tile_size / 2, 0);
      auto lower_right_x = std::min(center_x + tile_size / 2, width);
      auto lower_right_y = std::min(center_y + tile_size / 2, height);

      // one tile
      auto tile = Tile{std::make_unique<float[]>(tile_size * tile_size), Coord{center_x, center_y}, Coord{upper_left_x, upper_left_y}, Coord{lower_right_x, lower_right_y}};

      // fill array with floats
      for (int r = 0; r < tile_size; ++r) 
      {
        constexpr auto nbytes = tile_size * sizeof(float);
        auto offset = (upper_left_y + r) * width + upper_left_x;
        std::memcpy(tile.data.get() + r * tile_size, ptr + offset, nbytes);
      }

      tiles.emplace_back(std::move(tile));
      ++idx;
    }
  }
}