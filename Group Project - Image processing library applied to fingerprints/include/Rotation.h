/**
 * @file Rotation.h
 * @brief 
 * @version 0.1
 * @date 2023-01-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef ROTATION_H
#define ROTATION_H

#include "Transformation.h"

class Rotation : public Transformation {
    public:
        /**
         * @brief Construct a new Rotation object
         * 
         * @param theta : rotation angle in radians
         */
        Rotation(const double theta);

        
};

#endif