#ifndef DICTIONNARY_H
#define DICTIONNARY_H

#include <set>
//#include "Coordinates.h"
#include "Image.h"
#include "Mask.h"

class Dictionary {
    private:
        /**
         * @brief Set of patches
         */
        std::set<Image> dict;

        /**
         * @brief Number of patches and size of the patches in pixel
         */
        unsigned int N, size;

        /**
         * @brief Image the patches are generated from
         */
        Image image;

    public:
        /**
         * @brief Construct a new Dictionary object as a collection of random patches from the image
         * 
         * @param img : Image to crop
         * @param n : number of patches to generate
         * @param s : size of patches
         */
        Dictionary(const Image &img, const unsigned int n, const unsigned int s);

        /**
         * @brief Get the closest (wrt to similarity distance) patch
         * 
         * @param x : x coord of pixel to inpaint
         * @param y : y coord of pixel to inpaint
         * @param mask : boolean mask
         * @param img : image at this step
         * @return Image : closest patch
         */
        Image min_distance_patch(const unsigned int x, const unsigned int y, const Mask &mask, const Image &img) const;
};

#endif