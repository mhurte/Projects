/**
 * @file PixelCoordinates.h
 * @brief 
 * @version 0.1
 * @date 2023-01-09
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef PIXELCOORDINATES_H
#define PIXELCOORDINATES_H

#include "Image.h"
#include <vector>

class PixelCoordinates {
    private:
        std::vector<Eigen::Vector3d> coordinates;
        unsigned int N;
    public:
        /**
         * @brief Construct a new PixelCoordinates object from an existing Image
         * 
         * @param image : existing Image
         */
        PixelCoordinates(const Image &image);

        /**
         * @brief Construct a new PixelCoordinates object from coordinates
         * 
         * @param coords : coordinates of the pixels
         */
        PixelCoordinates(const std::vector<Eigen::Vector3d> &coords);

        /**
         * @brief Iterator at begin (const)
         * 
         * @return std::vector<Eigen::Vector3d>::const_iterator 
         */
        std::vector<Eigen::Vector3d>::const_iterator begin() const;

        /**
         * @brief Iterator at end (const)
         * 
         * @return std::vector<Eigen::Vector2d>::const_iterator 
         */
        std::vector<Eigen::Vector3d>::const_iterator end() const;

        /**
         * @brief Get the N attribute (number of pixels)
         * 
         * @return unsigned int 
         */
        unsigned int get_N() const;

        /**
         * @brief Get the ith pixel coordinate as an homogeneous vector
         * 
         * @param i : index
         * @return Eigen::Vector3d 
         */
        Eigen::Vector3d operator[](const unsigned int i) const;
};

#endif