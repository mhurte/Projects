#include "Translation.h"
#include <iostream>

Translation::Translation(const double tx, const double ty) : Transformation(Eigen::Matrix3d::Identity()) {
    data(0, 2) = tx;
    data(1, 2) = ty;

    std::cout << data << std::endl;
}