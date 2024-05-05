/**
 * @file inpainting.h
 * @brief 
 * @version 0.1
 * @date 2023-01-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef INPAINTING_H
#define INPAINTING_H

#include "Image.h"
#include "Dictionary.h"
#include "Mask.h"
#include <map>
#include <vector>
#include <array>
#include <cmath>

/**
 * @brief Inpaint a given pixel in an image based on a Dictionary of patches and a Mask
 * 
 * @param image : Image to inpaint
 * @param dict : Dictionary of square patches to draw from
 * @param x : x coord
 * @param y : y coord
 * @param mask : boolean Mask to use
 * @return Image 
 */
Image inpaint_pixel(Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, Mask &mask);

/**
 * @brief Inpaint a rectangular area in an image based on a Dictionary of patches and a Mask
 * 
 * @param image : Image to inpaint
 * @param dict : Dictionary of square patches to draw from
 * @param x : x coord of upper left corner
 * @param y : y coord of upper left corner
 * @param w : rectangle width
 * @param h : rectangle height
 * @return Image 
 */
Image inpaint_rectangle(const Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, const unsigned int w, const unsigned int h);

/**
 * @brief Inpaint a ring area in an image based on a Dictionary of patches and a Mask
 * 
 * @param image : Image to inpaint
 * @param dict : Dictionary of square patches to draw from
 * @param x : x coord of ring center
 * @param y : y coord of ring center
 * @param r : inner radius of the ring
 * @return Image 
 */
Image inpaint_ring(const Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, const unsigned int r);

/**
 * @brief Inpaint a masked are in an image based on a dictionary of patches
 * 
 * @param image : Image to inpaint
 * @param dict : Dictionary of square patches to draw from
 * @param mask : Mask in which to inpaint
 * @return Image 
 */
Image inpaint_mask(const Image &image, const Dictionary &dict, const Mask &mask);

#endif