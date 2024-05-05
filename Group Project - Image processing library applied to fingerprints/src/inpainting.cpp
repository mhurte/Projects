#include "inpainting.h"


Image inpaint_pixel(Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, Mask &mask) {
    // Get the most similar patch from the Dictionary
    Image patch = dict.min_distance_patch(x, y, mask, image);

    // Get its size in pixels and middle coordinate
    unsigned int size = patch.get_rows();
    unsigned int middle = std::ceil(size/2);

    // Inpaint the pixel at (x,y) using the middle pixel of the patch
    image.change_coeff_xy(x, y, patch(middle, middle));

    // Change the coefficient of the Mask to false where we inpainted
    mask.change_coeff_xy(x, y, false);

    return image;
}

Image inpaint_rectangle(const Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, const unsigned int w, const unsigned int h) {
    // Copy the image
    Image img(image);

    // Generate a mask of the right dimension, with a rectangle filled with true values
    Mask mask(img.get_cols(), img.get_rows(), x, y, w, h);

    // Inpaint every pixel in the rectangular area
    for (int i = x; i < x + w; i++) {
        for (int j = y; j < y + h; j++) {
            img = inpaint_pixel(img, dict, i, j, mask);
            //std::cout << i << ", " << j << std::endl;
        }
    }
    return img;
}

Image inpaint_ring(const Image &image, const Dictionary &dict, const unsigned int x, const unsigned int y, const unsigned int r) {
    // Copy the image
    Image img(image);

    // Generate a mask of the right dimension, with a disk filled of false values (true outside)
    Mask mask(img.get_cols(), img.get_rows(), x, y, r);

    // Map storing distance_to_center -> vector_of_points
    std::map<double, std::vector<std::array<int, 2>>> distances;

    // Fill the map with all pixels
    for (int i = 0; i < img.get_cols(); i++) {
        for (int j = 0; j < img.get_rows(); j++) {

            // >>> CHANGE THIS
            // Paint the ring on the image (as a check)
            if (mask(i,j))
                img.change_coeff_xy(i, j, 0.);

            // Compute the distance of the pixel to the center of the ring
            double distance = std::sqrt(std::pow(i - int(x), 2) + std::pow(j - int(y), 2));

            // If the pixel is in the ring
            if (distance >= r) {
                // Point (x,y)
                std::array<int, 2> point = {i,j};

                // Distance isn't yet in the map, insert it and create a vector of points
                if (distances.count(distance) == 0) {
                    //std::cout << distance << std::endl;
                    //std::cout << "Inserting vector 1" << std::endl;
                    std::vector<std::array<int, 2>> points_at_distance = {point};
                    //std::cout << "Inserting vector 2" << std::endl;
                    //std::cout << typeid(distance).name() << std::endl;
                    //std::cout << distances.count(distance) << std::endl;
                    /*for (auto p : points_at_distance) {
                        for (auto c : point) {
                            std::cout << c << ", ";
                        }
                        std::cout << std::endl;
                    }*/
                    distances[distance] = points_at_distance;
                    //std::cout << "Inserting vector 3" << std::endl;
                }
                // Insert the point in the vector
                else {
                    //std::cout << "Inserting point" << std::endl;
                    distances[distance].push_back(point);
                }
            }

        }
    }

    // >>> CHANGE THIS
    // Display the masked image
    std::cout << "Masked image, press a key to begin inpainting." << std::endl;
    img.display();
    //img.save_as("main_1_restauration_mask.png");

    std::cout << "Inpainting the image..." << std::endl;

    //int count = 0;

    // Iterate over the map (which is sorted by distance) and inpaint the pixels
    for (std::map<double, std::vector<std::array<int, 2>>>::const_iterator i = distances.begin(); i != distances.end(); i++) {
        // Get the vector of points at a given distance
        std::vector<std::array<int, 2>> points_at_distance = i->second;

        // Iterate over it
        for (std::vector<std::array<int, 2>>::const_iterator j = points_at_distance.begin(); j != points_at_distance.end(); j++) {
            // Inpaint each pixel 
            inpaint_pixel(img, dict, (*j)[0], (*j)[1], mask);
            
            /*
            count++;
            if (count%1000 == 0) {
                img.display();
                cv::waitKey(0);
            }*/
        }
    }

    return img;
}

Image inpaint_mask(const Image &image, const Dictionary &dict, const Mask &mask) {
    // Copy the image
    Image img(image);
    Mask mask2(mask);

    // Map storing distance_to_center -> vector_of_points
    std::map<double, std::vector<std::array<int, 2>>> distances;

    // USE A BARYCENTER FUNCTION INSTEAD
    int X = 0;
    int Y = 0;
    int w = 0;
    for (int x = 0; x < mask.get_cols(); x++) {
        for (int y = 0; y < mask.get_rows(); y++) {
            if (!mask(x, y)) { //Pixel is in the kept area of the Image
                X += x;
                Y += y;
                w += 1;
            }
        }
    }
    X /= w;
    Y /= w;

    // Fill the map with all pixels
    for (int i = 0; i < img.get_cols(); i++) {
        for (int j = 0; j < img.get_rows(); j++) {

            // Paint the mask on the image (as a check)
            if (mask(i,j))
                img.change_coeff_xy(i, j, 0.);

            // Compute the distance of the pixel to the center of the ring
            double distance = std::sqrt(std::pow(i - int(X), 2) + std::pow(j - int(Y), 2));

            // If the pixel is in the masked area
            if (mask(i, j)) {
                // Point (x,y)
                std::array<int, 2> point = {i,j};

                // Distance isn't yet in the map, insert it and create a vector of points
                if (distances.count(distance) == 0) {
                    //std::cout << distance << std::endl;
                    //std::cout << "Inserting vector 1" << std::endl;
                    std::vector<std::array<int, 2>> points_at_distance = {point};
                    //std::cout << "Inserting vector 2" << std::endl;
                    //std::cout << typeid(distance).name() << std::endl;
                    //std::cout << distances.count(distance) << std::endl;
                    /*for (auto p : points_at_distance) {
                        for (auto c : point) {
                            std::cout << c << ", ";
                        }
                        std::cout << std::endl;
                    }*/
                    distances[distance] = points_at_distance;
                    //std::cout << "Inserting vector 3" << std::endl;
                }
                // Insert the point in the vector
                else {
                    //std::cout << "Inserting point" << std::endl;
                    distances[distance].push_back(point);
                }
            }

        }
    }

    // >>> CHANGE THIS
    // Display the masked image
    //std::cout << "Masked image, press a key to begin inpainting." << std::endl;
    //img.display();
    //img.save_as("main_1_restauration_mask.png");

    std::cout << "Inpainting the image..." << std::endl;

    //int count = 0;

    // Iterate over the map (which is sorted by distance) and inpaint the pixels
    for (std::map<double, std::vector<std::array<int, 2>>>::const_iterator i = distances.begin(); i != distances.end(); i++) {
        // Get the vector of points at a given distance
        std::vector<std::array<int, 2>> points_at_distance = i->second;

        // Iterate over it
        for (std::vector<std::array<int, 2>>::const_iterator j = points_at_distance.begin(); j != points_at_distance.end(); j++) {
            // Inpaint each pixel 
            inpaint_pixel(img, dict, (*j)[0], (*j)[1], mask2);
            
            /*
            count++;
            if (count%1000 == 0) {
                img.display();
                cv::waitKey(0);
            }*/
        }
    }

    return img;
}