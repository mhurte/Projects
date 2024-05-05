/**
 * @file local_warp.h
 * @brief 
 * @version 0.1
 * @date 2023-01-12
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#ifndef LOCAL_WARP_H
#define LOCAL_WARP_H

/**
 * @brief TEMPORARY function computing local translation factors along x axis
 * 
 * @param x : coord
 * @param y : coord
 * @param x_center : center of the warp
 * @param y_center : center of the warp
 * @param a : coef
 * @param b : coef
 * @param c : coef
 * @param d : coef
 * @return double 
 */
double delta_x(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d);

/**
 * @brief TEMPORARY function computing local translation factors along x axis
 * 
 * @param x : coord
 * @param y : coord
 * @param x_center : center of the warp
 * @param y_center : center of the warp
 * @param a : coef
 * @param b : coef
 * @param c : coef
 * @param d : coef
 * @return double 
 */
double delta_y(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d);

#endif