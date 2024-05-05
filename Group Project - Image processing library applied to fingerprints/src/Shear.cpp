#include "Shear.h"

Shear::Shear(const double s, const bool horizontal) : Transformation(Eigen::Matrix3d::Identity()) {
    if (horizontal)
        data(0, 1) = s;
    else
        data(1, 0) = s;
}