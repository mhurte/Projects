#include "Dictionary.h"
#include <cstdlib>

Dictionary::Dictionary(const Image &img, const unsigned int n, const unsigned int s = 9) {
    if (s%2 == 0)
        std::cerr << "Patch size " << s << " is not an odd integer." << std::endl;
    else {
        N = n;
        size = s;
        image = img;
        for (int i = 0; i < N; i++) {
            // Generate random coordinates in the image (reduced by the size of the patch on the right and bottom)
            unsigned int x = rand() % (img.get_cols() - 1 - size);
            unsigned int y = rand() % (img.get_rows() - 1 - size);

            // Generate a patch by cropping the image
            Image patch = image.crop(x, y, size, size);

            // Insert it
            dict.insert(patch);
        }
    }
}

Image Dictionary::min_distance_patch(const unsigned int x, const unsigned int y, const Mask &mask, const Image &img) const {
    // Initialize a huge minimal distance
    double min_distance = 1e6;

    // Declare a patch object (return)
    Image patch(size, size, "best_patch");

    // Number of pixels to reach the side of the patch from its center
    unsigned int half_size = std::floor(size/2);

    // Used to check if we are in the right area of the mask
    bool found_pixel = false;

    // To estimate the complexity reduction
    int pixel_counter = 0;

    // Iterate over the "dictionary" of patches
    for (std::set<Image>::const_iterator i = dict.begin(); i != dict.end(); i++) {
        //i->display();
        //cv::waitKey(0);

        // Accumulator for distance computation
        double squares_sum = 0.;

        // For every pixel in the patch
        for (int j = 0; j < size; j++) {
            int k = 0;
            while (squares_sum < min_distance && k < size) {
            //for (int k = 0; k < size; k++) {
                // Coordinates in the whole image and mask
                unsigned int X = x + j - half_size;
                unsigned int Y = y + k - half_size;

                // Bounds check
                if (X >= 0 && X < image.get_cols() && Y >= 0 && Y < image.get_rows()) {
                    //Only consider pixels that are unmasked in the image
                    if (!mask(X,Y)) {
                        // Found at least one unmasked pixel
                        found_pixel = true;
                        // Add the squared distance
                        squares_sum += pow((*i)(j,k) - img(X,Y), 2);

                        pixel_counter++;
                    }
                    //std::cout << std::endl;
                }
            k++;
            }
        }

        // Euclidean similarity-distance
        //double distance = squares_sum;

        // If smaller, keep it and this patch
        if (squares_sum < min_distance) {
            min_distance = squares_sum;
            patch = *i;
        }
    }

    int total_pixels = N * size * size;

    //std::cout << "Processed " << pixel_counter << " pixels, reduced from " << total_pixels << " (" << 100. * pixel_counter/total_pixels << "%)" << std::endl;

    if (found_pixel)
        return patch;
    else {
        std::cerr << "No unmasked pixel in the patch! Returning white patch." << std::endl;
        return Image(size, size, 1., "white_patch.png");
    }
        
}