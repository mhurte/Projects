/**
 * @file Translation.h
 * @brief 
 * @version 0.1
 * @date 2023-01-16
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef TRANSLATION_H
#define TRANSLATION_H

#include "Transformation.h"

class Translation : public Transformation {
    public:
        /**
         * @brief Construct a new Translation object
         * 
         * @param tx : translation along x axis
         * @param ty : translation along y axis
         */
        Translation(const double tx, const double ty);

        
};

#endif