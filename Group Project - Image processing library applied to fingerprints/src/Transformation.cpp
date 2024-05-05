#include "Transformation.h"

Transformation::Transformation() : Matrix<double>(Eigen::Matrix3d::Identity()) {}

Transformation::Transformation(const Eigen::Matrix<double, 3, 3> &mat) : Matrix<double>(mat) {}
        
PixelCoordinates Transformation::operator*(const PixelCoordinates &coords) const {
    std::vector<Eigen::Vector3d> vec;
    //vec.reserve(coords.get_N());
    for (std::vector<Eigen::Vector3d>::const_iterator i = coords.begin(); i != coords.end(); i++) {
        vec.push_back(data * (*i));
        //std::cout << data * (*i) << '\n' << std::endl;
    }
    return PixelCoordinates(vec);
}

Transformation Transformation::operator*(const Transformation &right) const {
    return Transformation(data * right.data);
}

Transformation Transformation::inverse() const {
    //std::cout << data.inverse() * data << std::endl;
    return Transformation(data.inverse());
}