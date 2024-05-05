/**
 * @file Pixel.cpp
 * @brief Implements a pixel class (container for coordinates)
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Pixel.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <cmath>
#include <iostream>


Pixel::Pixel(int i_x, int i_y) {
    x = i_x;
    y = i_y;
}


Pixel Pixel::change_coord(int i_x, int i_y) const {
    Pixel o_p(0, 0);
    o_p.x = i_x;
    o_p.y = i_y;
    return o_p;
}


float Pixel::distance(const Pixel& i_other) const {
    //Euclidian norm
    float o_distance = std::sqrt(std::pow((this->x - i_other.x), 2) + std::pow((this->y - i_other.y), 2));
    return o_distance;
}


Pixel Pixel::operator+(const Pixel& i_other) const {
    Pixel o_sum = Pixel(i_other.x + x, i_other.y + y);
    return o_sum;
}


Pixel Pixel::operator-(const Pixel& i_other) const {
    Pixel o_vector = Pixel(i_other.x - x, i_other.y - y);
    return o_vector;
}


float Pixel::length() const {
    //The norm is the distance from the origin (vector space context)
    Pixel origin_pixel = Pixel(0,0);
    float o_length = distance(origin_pixel);
    return o_length;
}


float Pixel::inner_product(const Pixel& i_other) const {
    //Sum of the products coordinate by coordinate
    float o_inner_product = x * i_other.x + y * i_other.y;
    return o_inner_product;
}


float Pixel::angle_cos(const Pixel& i_other, const Pixel& i_centre) const {
    Pixel vector_1 = *this - i_centre;
    Pixel vector_2 = i_other - i_centre;
    float o_cos = vector_1.inner_product(vector_2) / (vector_1.length() * vector_2.length());
    return o_cos;
}
