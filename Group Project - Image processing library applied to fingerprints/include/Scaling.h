/**
 * @file Scaling.h
 * @brief 
 * @version 0.1
 * @date 2023-01-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef SCALING_H
#define SCALING_H

#include "Transformation.h"

class Scaling : public Transformation {
    public:
        /**
         * @brief Construct a new Scaling object
         * 
         * @param sx : scaling factor along x axis
         * @param sy : scaling factor along y axis
         */
        Scaling(const double sx, const double sy);

        /**
         * @brief Construct a new Scaling object
         * 
         * @param s : scaling factor along both axes
         */
        Scaling(const double s);
        
};

#endif