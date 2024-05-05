/**
 * @file transform.h
 * @brief 
 * @version 0.1
 * @date 2023-01-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef TRANSFORM_H
#define TRANSFORM_H

#include "Image.h"
#include "PixelCoordinates.h"
#include "Transformation.h"
#include <algorithm>

/**
 * @brief Build an Image from initial coordinates and pixel values and output coordinates (after transformation) by rounding the pixel coordinates (floor)
 * 
 * @param input_coords : pixel coordinates before transformation
 * @param image : pixel values in matrix format
 * @param output_coords : pixel coordinates after transformation
 * @return Image 
 */
Image nearest_neighbor(const PixelCoordinates &input_coords, const Image &image, const PixelCoordinates &output_coords);

/**
 * @brief Comparison function for pairs (squared distance, index)
 * 
 * @param p1 : first pair
 * @param p2 : second pair
 * @return true
 * @return false 
 */
bool compare_pair(const std::pair<double, int> &p1, const std::pair<double, int> &p2);

/**
 * @brief Apply a Transformation to an Image. Pixel values are computing using the inverse transformation and first order interpolation.
 * 
 * @param input_coords : inverse transformed pixel coordinates (on the input image)
 * @param pixel_values : input image, containing the pixel values
 * @param output_coords : coordinates of the pixels on the output image
 * @return Image 
 */
Image interpolate_1st_inverse(const PixelCoordinates &input_coords, const Image &pixel_values, const PixelCoordinates &output_coords);

/**
 * @brief Apply a Transformation to an Image. Pixel values are computing using the inverse transformation and first order interpolation.
 * 
 * @param image : Image to transform
 * @param transfo : Transformation to apply
 * @return Image 
 */
Image interpolate_1st_inverse(const Image &image, const Transformation &transfo);

/**
 * @brief Apply a Transformation to an Image. Interpolates pixel values using bicubic interpolation and the inverse transformation.
 * 
 * @param image 
 * @param transfo 
 * @return Image 
 */
Image interpolate_bicubic(const Image &image, const Transformation &transfo);

/**
 * @brief Compute the affine transformation based on 3 control points
 * 
 * @param in : coordinates of the 3 pixels on the input image
 * @param out : coordinates of the 3 pixels on the output image
 * @return Transformation 
 */
Transformation transformation_between(const PixelCoordinates &in, const PixelCoordinates &out);



#endif