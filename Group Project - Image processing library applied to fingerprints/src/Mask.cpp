#include "Mask.h"
#include <cmath>

Mask::Mask(const unsigned int cols, const unsigned int rows, const bool val) : Matrix<bool>(cols, rows, val) {}

Mask::Mask(const unsigned int cols, const unsigned int rows, const unsigned int x, const unsigned int y, const unsigned int w, const unsigned int h) 
    : Matrix<bool>(cols, rows, false){
    for (int i = x; i <= x + w; i++) {
        for (int j = y; j <= y + h; j++) {
            data(j,i) = true;
        }
    }
}

Mask::Mask(const unsigned int cols, const unsigned int rows, const unsigned int x, const unsigned int y, const unsigned int r)
    : Matrix<bool>(cols, rows, false){
    for (int i = 0; i < cols; i++) {
        for (int j = 0; j < rows; j++) {
            if (std::sqrt(std::pow(i - int(x), 2) + std::pow(j - int(y), 2)) >= r)
                data(j,i) = true;
        }
    }
}

Mask Mask::invert() const {
    Mask mask(*this);
    for (int x = 0; x < get_cols(); x++)
        for (int y =0; y < get_rows(); y++)
            mask.change_coeff_xy(x, y, !(*this)(x, y));

    return mask;
}

Mask Mask::logical_and(const Mask &mask2) const {
    Mask mask1(get_cols(), get_rows(), false);

    for (int x = 0; x < get_cols(); x++)
        for (int y =0; y < get_rows(); y++)
            mask1.change_coeff_xy(x, y, (*this)(x, y) && mask2(x, y));

    return mask1;
}

Mask Mask::logical_nand(const Mask &mask2) const {
    Mask mask1(get_cols(), get_rows(), false);

    for (int x = 0; x < get_cols(); x++)
        for (int y =0; y < get_rows(); y++)
            mask1.change_coeff_xy(x, y, !((*this)(x, y) && mask2(x, y)));

    return mask1;
}

Mask Mask::logical_or(const Mask &mask2) const {
    Mask mask1(get_cols(), get_rows(), false);

    for (int x = 0; x < get_cols(); x++)
        for (int y =0; y < get_rows(); y++)
            mask1.change_coeff_xy(x, y, (*this)(x, y) || mask2(x, y));

    return mask1;
}

Mask Mask::operator!() const {
    Mask mask(get_cols(), get_rows(), false);

    for (int x = 0; x < get_cols(); x++)
        for (int y =0; y < get_rows(); y++)
            mask.change_coeff_xy(x, y, !(*this)(x, y));

    return mask; 
}

Mask Mask::fill_holes(const bool val) const {
    // Create a copy of the Mask
    Mask mask(*this);

    // Iterate over the columns and get the up- and down-most non-white pixels
    for (int x = 0; x < get_cols(); x++) {
        // Up coord
        int up_y = 0;
        bool found_up = false;
        while (!found_up && up_y < get_rows()) {
            if ((*this)(x, up_y) == val) {
                found_up = true;
            }
            up_y++;
        }

        // Down coord
        int down_y = get_rows() - 1;
        bool found_down = false;
        while (!found_down && down_y >= 0) {
            if ((*this)(x, down_y) == val) {
                found_down = true;
            }
            down_y--;
        }   

        // Fill the mask between these two coordinates
        for (int y = up_y; y <= down_y; y++) {
            mask.change_coeff_xy(x, y, val);
        }     
    }

    // Iterate over the rows and get the left- and right-most non-white pixels
    for (int y = 0; y < get_rows(); y++) {
        // Left coord
        int left_x = 0;
        bool found_left = false;
        while (!found_left && left_x < get_cols()) {
            if ((*this)(left_x, y) == val) {
                found_left = true;
            }
            left_x++;
        }

        // Right coord
        int right_x = get_cols() - 1;
        bool found_right = false;
        while (!found_right && right_x >= 0) {
            if ((*this)(right_x, y) == val) {
                found_right = true;
            }
            right_x--;
        }   

        // Fill the mask between these two coordinates
        for (int x = left_x; x <= right_x; x++) {
            mask.change_coeff_xy(x, y, val);
        }     
    }

    return mask;
}

Mask Mask::shrink(const int pixels, const bool inside) const {
    // Create a copy of the Mask
    Mask mask(*this);

    // Iterate over the columns and get the boundaries of the inside-valued area
    for (int x = 0; x < get_cols(); x++) {
        int y_left = get_cols();
        int y_right = 0;
        for (int y = 0; y < get_rows(); y++) {
            if ((y == 0 && (*this)(x, y) == inside) || (y > 0 && (*this)(x, y - 1) != inside && (*this)(x, y) == inside))
                y_left = y;

            if ((y == get_rows() - 1 && (*this)(x, y) == inside) || (y < get_rows() && (*this)(x, y + 1) != inside && (*this)(x, y) == inside))
                y_right = y;
        }

        // Shrink the mask vertically
        for (int y = 0; y < get_rows(); y++) {
            if (y <= y_left + pixels || y >= y_right - pixels)
                mask.change_coeff_xy(x, y, !inside);
        }
    }

    // Iterate over the rows and get the boundaries of the inside-valued area
    for (int y = 0; y < get_rows(); y++) {
        int x_up = get_rows();
        int x_down = 0;
        for (int x = 0; x < get_cols(); x++) {
            if ((x == 0 && (*this)(x, y) == inside) || (x > 0 && (*this)(x - 1, y) != inside && (*this)(x, y) == inside))
                x_up = x;

            if ((x == get_cols() - 1 && (*this)(x, y) == inside) || (x < get_cols() && (*this)(x + 1, y) != inside && (*this)(x, y) == inside))
                x_down = x;
        }

        // Shrink the mask vertically
        for (int x = 0; x < get_cols(); x++) {
            if (x <= x_up + pixels || x >= x_down - pixels)
                mask.change_coeff_xy(x, y, !inside);
        }
    }

    return mask;
}