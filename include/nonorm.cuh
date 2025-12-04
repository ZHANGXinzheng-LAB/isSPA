#pragma once

#include <memory>
#include <ostream>
#include <string>

#include "DataReader2.h"
#include "emdata.h"
#include "templates.hpp"
#include "tileimages.hpp"

struct SearchNoNorm 
{
  Parameters para;
  EulerData euler;

  struct Size 
  {
    size_t width;
    size_t height;
  };

  struct impl;
  std::unique_ptr<impl> pimpl;

  size_t padded_template_size;
  size_t batch_size;
  int padding_size;
  int overlap;
  int nx, ny;
  int block_x, block_y;
  int line_count;
  float phi_step;
  bool invert, phase_flip;

  void LoadTemplate(const Templates & temp);
  void LoadImage(const TileImages::Tile & tile);
  void SetParams(const TileImages::Params & params);
  void PreprocessImage();
  void SplitImage();
  void PreprocessTemplate();
  void PickParticles(std::vector<float>& scores, float euler3);
  void OutputScore(std::string & output, float euler3, std::vector<float>& scores, const TileImages & tiles, const TileImages::Tile& tile);

  void work(const Templates & temp, const TileImages & tiles, std::string & output);
  void work_verbose(const Templates & temp, const TileImages & tiles, std::string & output);

  SearchNoNorm(const Config & c, const EulerData & e, Size img, int device = 0);
  ~SearchNoNorm();
};