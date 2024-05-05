#include "Matrix.hpp"

float distance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2) {
  return std::sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}
