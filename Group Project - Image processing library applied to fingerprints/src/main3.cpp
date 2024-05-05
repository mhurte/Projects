/**
 * @file main3.cpp
 * @brief 
 * @version 0.1
 * @date 2023-01-11
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Image.h"
#include <cmath>

double delta_x(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d) {
    const int X = x - x_center;
    const int Y = y - y_center;
    return a * std::sin(b * X) * std::exp(-c * X * X - d * Y * Y);
}

double delta_y(const int x, const int y, const int x_center, const int y_center, double a, double b, double c, double d) {
    const int X = x - x_center;
    const int Y = y - y_center;
    return a * std::sin(b * Y) * std::exp(-c * X * X - d * Y * Y);
}

int main() {
    Image image("data/clean_finger.png");

    Image img(image.get_cols(), image.get_rows(), 1., "");

    for (int x = 0; x < image.get_cols(); x++) {
        for (int y = 0; y < image.get_rows(); y++) {
            double x_out = x + delta_x(x, y, image.get_cols()/2, 3*image.get_rows()/4, 1000., 0.001, 0.001, 0.001);
            double y_out = y + delta_y(x, y, image.get_cols()/2, 3*image.get_rows()/4, 1000., 0.001, 0.001, 0.001);

            if (x_out > 0 && x_out < img.get_cols() - 1 && y_out > 0 && y_out < img.get_rows() - 1)
                img.change_coeff_xy(x, y, image(std::floor(x_out), std::floor(y_out)));
        }
    }

    img.display();

    return 0;
}
