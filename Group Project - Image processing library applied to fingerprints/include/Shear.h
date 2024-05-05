/**
 * @file Shear.h
 * @brief 
 * @version 0.1
 * @date 2023-01-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef SHEAR_H
#define SHEAR_H

#include "Transformation.h"

class Shear : public Transformation {
    public:
        /**
         * @brief Construct a new Shear object
         * 
         * @param s : shear factor
         * @param horizontal : true for horizontal, false for vertical
         */
        Shear(const double s, const bool horizontal);
        
};

#endif