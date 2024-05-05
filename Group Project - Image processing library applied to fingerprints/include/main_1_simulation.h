/**
 * @file main_1_simulation.h
 * @brief Implements methods developped in main 1 simulation
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef MAIN_1_SIMULATION_H
#define MAIN_1_SIMULATION_H

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "Pixel.h"
#include "Image.h"

/**
 * @brief mathematical function c(r) = e^{-r} which tends to 0 while r tends to infinity and c(0)=1.
 * @param i_r : radial distance from the center to the considered Pixel.
 */
float c1(float i_r);


/**
 * @brief mathematical function c(r) = 1/(r+1)^2 which tends to 0 while r tends to infinity and c(0)=1.
 * @param i_r : radial distance from the center to the considered Pixel.
 */
float c2(float i_r);


/**
 * @brief mathematical function c(r) = 1/(log(1_r)+1) which tends to 0 while r tends to infinity and c(0)=1.
 * @param i_r : radial distance from the center to the considered Pixel.
 */
float c3(float i_r);


/**
 * @brief mathematical function c(r) = 0.5*(1+tanh(-r)) which tends to 0 while r tends to infinity and c(0)=1.
 * @param i_r : radial distance from the center to the considered Pixel.
 */
float c4(float i_r);


/**
 * @brief the isotropic coefficient 
 * @param i_norm Length of diagnoal of the input image to normalize.
 * @param i_Pixel : input Pixel
 * @param i_Pixel_centre : pressure centre
 * @param  i_cfun: index of the c-funtion to use
 */
float coeff(Pixel i_Pixel_1, Pixel i_Pixel_2, int i_cfun, float i_norm);


/**
 * @brief the anisotropic coefficient 
 * 
 * @param i_Pixel Input image.
 * @param i_Pixel_centre Coordinates of the place where the pressure is considered maximal.
 * @param i_Pixel_direction : pressure direction
 * @param  i_cfun: imply which cfuntion to use
 */
float coeff_anisotropic(Pixel i_Pixel, Pixel i_Pixel_centre, Pixel i_direction, float i_norm,int i_cfun);


/**
 * @brief the isotropic result of pressure variation
 * 
 * @param i_image Input image.
 * @param i_centre Coordinates of the place where the pressure is considered maximal.
 * @param i_cfun Index of the c-funtion to use.
 */
Image isotropic(const Image &i_image, Pixel i_centre, int i_cfun);


/**
 * @brief the anisotropic result of pressure variation
 * 
 * @param i_image Input image.
 * @param i_centre Coordinates of the place where the pressure is considered maximal.
 * @param i_centre : pressure direction
 * @param  i_cfun: imply which cfuntion to use
 */
Image anisotropic(const Image &i_image, Pixel i_centre, Pixel i_direction, int i_cfun);


#endif