/**
 * @file Transformation.h
 * @brief 
 * @version 0.1
 * @date 2023-01-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef TRANSFORMATION_H
#define TRANSFORMATION_H

#include "Matrix.hpp"
#include "PixelCoordinates.h"
#include <vector>

class Transformation : public Matrix<double> {
    public:
        /**
         * @brief Construct a new Transformation object as the identity matrix
         * 
         */
        Transformation();

        /**
         * @brief Construct a new Transformation object from an Eigen::Matrix
         * 
         * @param mat : matrix
         */
        Transformation(const Eigen::Matrix<double, 3, 3> &mat);

        /**
         * @brief Apply the transformation to some PixelCoordinates
         * 
         * @param coords : homogenous coordinates of the pixels to transform
         * @return PixelCoordinates 
         */
        virtual PixelCoordinates operator*(const PixelCoordinates &coords) const;

        /**
         * @brief Compose transformations (using matrix multiplication)
         * 
         * @param right : right side matrix in multiplication 
         * @return Transformation 
         */
        Transformation operator*(const Transformation &right) const;

        //Image operator*(const Image &img) const;

        /**
         * @brief Inverse the transformation
         * 
         * @return Transformation 
         */
        Transformation inverse() const;
};

#endif