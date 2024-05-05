/**
 * @file Pixel.h
 * @brief Implements a pixel class (container for coordinates)
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef PIXEL_H
#define PIXEL_H

class Pixel {
    private:
        int x; //height coordinate
        int y; //line coordinate
    public:
        /**
         * \brief Constructor of the class.
         * \param i_x Wanted x coordinate.
         * \param i_y Wanted y coordinate.
        */
        Pixel(int i_x, int i_y);


        /**
         * \brief Change coordinates of a pixel (in a copy).
         * \param i_x New x coordinate.
         * \param i_y New y coordinate.
         * \return Modified copy of the original Pixel.
        */
        Pixel change_coord(int i_x, int i_y) const;


        /**
         * \brief Computes euclidian distance between Pixels.
         * \param i_other The pixel to which compute the distance.
         * \return Computed distance.
        */
        float distance(const Pixel& i_other) const;


        /**
         * \brief Overload of + for pixels.
         *
         * Addition of pixel is performed as on vectors, coordinate by coordinate.
         *
         * \param i_other The pixel to add.
         * \return Result of the sum.
        */
        Pixel operator+(const Pixel& i_other) const;


        /**
         * \brief Overload of - for pixels.
         *
         * Subtraction of pixel is performed as on vectors, coordinate by coordinate.
         *
         * \param i_other The pixel to subtract.
         * \return Result of the subtraction.
        */
        Pixel operator-(const Pixel& i_other) const;

        /**
         * \brief Compute the euclidian norm of a pixel seen as a vector.
         * \return Norm of the pixel.
        */
        float length() const;

        /**
         * \brief Compute scalar product on pixels seen as vectors.
         * \param i_other The other pixel to compute the inner product.
         * \return Result of the inner product.
        */
        float inner_product(const Pixel& i_other) const;

        /**
         * \brief Compute the cosine of the angle between two pixels wrt a given center.
         * \param i_other First pixel.
         * \param i_center Considered center to compute the angle around.
         * \return Cosine of the formed angle.
        */
        float angle_cos(const Pixel& i_other, const Pixel& i_centre) const;
};

#endif
