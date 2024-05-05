#include "local_warp.h"
#include <cmath>

double delta_x(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d) {
    const int X = x - x_center;
    const int Y = y - y_center;
    return a * std::sin(b * X) * std::exp(-c * X * X - d * Y * Y);
}

double delta_y(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d) {
    const int X = x - x_center;
    const int Y = y - y_center;
    return a * std::sin(b * Y) * std::exp(-c * X * X - d * Y * Y);
}