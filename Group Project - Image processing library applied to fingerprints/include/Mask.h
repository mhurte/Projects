#ifndef MASK_H
#define MASK_H

#include "Matrix.hpp"
//#include "Image.h"

class Mask : public Matrix<bool> {
    public:
        /**
         * @brief Construct a new Mask object filled with a value
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param val : value to input everywhere
         */
        Mask(const unsigned int cols, const unsigned int rows, const bool val);

        /**
         * @brief Construct a new rectangular Mask object
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param x : upper left corner of rectangle x coord
         * @param y : upper left corner of rectangle y coord
         * @param w : width of rectangle
         * @param h : height of rectangle
         */
        Mask(const unsigned int cols, const unsigned int rows, const unsigned int x, const unsigned int y, const unsigned int w, const unsigned int h);

        /**
         * @brief Construct a new ring shaped Mask object
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param x : center x coord
         * @param y : center y coord
         * @param r : inner radius of the ring
         */
        Mask(const unsigned int cols, const unsigned int rows, const unsigned int x, const unsigned int y, const unsigned int r);

        /**
         * @brief Invert the boolean values of the mask
         * 
         * @return Mask 
         */
        Mask invert() const;

        /**
         * @brief Perform a coefficient-wise logical and operation.
         * 
         * @param mask : second Mask
         * @return Mask 
         */
        Mask logical_and(const Mask &mask) const;

        /**
         * @brief Perform a coefficient-wise logical nand operation.
         * 
         * @param mask : second Mask
         * @return Mask 
         */
        Mask logical_nand(const Mask &mask) const;

        /**
         * @brief Perform a coefficient-wise logical or operation.
         * 
         * @param mask : second Mask
         * @return Mask 
         */
        Mask logical_or(const Mask &mask) const;

        /**
         * @brief Overload the negation operator (coefficient-wise)
         * 
         * @return Mask 
         */
        Mask operator!() const;

        /**
         * @brief Fill the holes between values val in the Mask
         * 
         * @param val : value to fill in between and with
         * @return Mask 
         */
        Mask fill_holes(const bool val) const;

        /**
         * @brief Shrink the mask by a certain number of pixels on each side
         * 
         * @param pixels : number of pixels to shrink by on each side
         * @param inside : boolean value in the area to be shrinked
         * @return Mask 
         */
        Mask shrink(const int pixels, const bool inside = false) const;
};

#endif