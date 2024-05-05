#include "Rotation.h"
#include <cmath>

Rotation::Rotation(const double theta) : Transformation(Eigen::Matrix3d::Zero()) {
    data(0, 0) = std::cos(theta);
    data(0, 1) = std::sin(theta);
    data(1, 0) = -std::sin(theta);
    data(1, 1) = std::cos(theta);
    data(2, 2) = 1.;
}