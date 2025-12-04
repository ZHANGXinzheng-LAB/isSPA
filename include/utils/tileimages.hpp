#pragma once

#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "DataReader2.h"

struct TileImages 
{
  struct Coord 
  {
    int x;
    int y;
  };

  struct Tile 
  {
    std::unique_ptr<float[]> data;
    Coord center;
    Coord upper_left;
    Coord lower_right;
  };

  struct Params 
  {
    double defocus;
    double dfdiff;
    double dfang;
    int width;
    int height;
  };

  class iterator : public std::iterator<std::output_iterator_tag, Tile> 
  {
    private:
    std::vector<Tile>::iterator it;

    public:
    explicit iterator(std::vector<Tile>::iterator i) : it(i) {}
    iterator& operator++() 
    {
      ++it;
      return *this;
    }
    iterator operator++(int) 
    {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(iterator other) const { return it == other.it; }
    bool operator!=(iterator other) const { return !(*this == other); }
    reference operator*() const { return *it; }
  };

  class const_iterator : public std::iterator<std::output_iterator_tag, Tile, std::ptrdiff_t,
                                              const Tile*, const Tile&> {
    private:
    std::vector<Tile>::const_iterator it;

    public:
    explicit const_iterator(std::vector<Tile>::const_iterator i) : it(i) {}
    const_iterator& operator++() 
    {
      ++it;
      return *this;
    }
    const_iterator operator++(int) {
      const_iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(const_iterator other) const { return it == other.it; }
    bool operator!=(const_iterator other) const { return !(*this == other); }
    reference operator*() const { return *it; }
  };

  iterator begin() { return iterator(tiles.begin()); }
  iterator end() { return iterator(tiles.end()); }
  const_iterator begin() const { return const_iterator(tiles.begin()); }
  const_iterator end() const { return const_iterator(tiles.end()); }

  // constructor
  TileImages(const LST::Entry& input);

  std::vector<Tile> tiles;
  std::string rpath;
  int unused;
  Params p;
};