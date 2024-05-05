#include "PixelCoordinates.h"

PixelCoordinates::PixelCoordinates(const Image &image) {
    N = image.get_cols() * image.get_rows();
    //coordinates.reserve(N);
    for (double x = 0.; x < image.get_cols(); x++) {
        for (double y = 0.; y < image.get_rows(); y++) {
            Eigen::Vector3d vec = {x, y, 1.};
            coordinates.push_back(vec);
        }
    }
}

PixelCoordinates::PixelCoordinates(const std::vector<Eigen::Vector3d> &coords) {
    N = coords.size();
    coordinates = coords;
}

std::vector<Eigen::Vector3d>::const_iterator PixelCoordinates::begin() const {
    return coordinates.begin();
}

std::vector<Eigen::Vector3d>::const_iterator PixelCoordinates::end() const {
    return coordinates.end();
}

Eigen::Vector3d PixelCoordinates::operator[](const unsigned int i) const {
    return coordinates[i];
}

unsigned int PixelCoordinates::get_N() const {
    return N;
}