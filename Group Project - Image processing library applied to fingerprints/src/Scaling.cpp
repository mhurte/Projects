#include "Scaling.h"

Scaling::Scaling(const double sx, const double sy) : Transformation(Eigen::Matrix3d::Zero()) {
    data(0, 0) = sx;
    data(1, 1) = sy;
    data(2, 2) = 1.;
}

Scaling::Scaling(const double s) : Transformation(Eigen::Matrix3d::Zero()) {
    data(0, 0) = s;
    data(1, 1) = s;
    data(2, 2) = 1.;
}