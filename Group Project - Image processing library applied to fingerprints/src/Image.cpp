/**
 * @file Image.cpp
 * @brief 
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Image.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <string>
#include <algorithm>
#include <iostream>
#include <cmath>

Image::Image() : Matrix<double>::Matrix() {
    filename = "";
}

Image::Image(std::string f) : Matrix<double>::Matrix() {
    filename = f;
    cv::Mat image = cv::imread(f, cv::IMREAD_GRAYSCALE);
    cv::cv2eigen(image, data);

    data /= 255.;

    std::cout << "Imported image " << filename << " of size (" << data.cols() << ", " << data.rows() << ")" << std::endl;
}

Image::Image(const unsigned int cols, const unsigned int rows, 
             std::string f = "") : Matrix<double>::Matrix(cols, rows) {
    filename = f;
}

Image::Image(const Image &i) : Matrix<double>() {
    data = i.data;
    filename = i.filename;
}

Image::Image(const Mask &mask) : Matrix<double>::Matrix(mask.get_cols(), mask.get_rows()) {
    for (int x = 0; x < get_cols(); x++)
        for (int y = 0; y < get_rows(); y++)
            change_coeff_xy(x, y, mask(x, y) ? 1. : 0.);
    filename = "image_from_mask";
}

Image::Image(const unsigned int cols, const unsigned int rows, const double val, 
             std::string f = "") : Matrix<double>::Matrix(cols, rows, val) {
    assert((0. <= val) && (val <= 1.));
    filename = f;
}

Image::Image(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string f = "") : Matrix<double>::Matrix(m) {
    filename = f;
}

Image::Image(const Matrix<double> &m, const std::string f) : Matrix<double>::Matrix(m) {
    filename = f;
}


Image::Image(const unsigned int cols, const unsigned int rows, const unsigned int options) : Matrix<double>::Matrix(cols, rows, 0.0) {
    // Extracting options
    int u = (options % 10) %4;
    int tens = (((options % 100) - u) / 10 ) % 8;

    //Type of filling
    switch (u) {
        // Flat black
        case 0:
            break;
        // Flat gray
        case 1:
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    data(i,j) = 0.5;
                }
            }
            break;
        // Flat white
        case 2:
            for (int j = 0; j < cols; j++) {
                for (int i = 0; i < rows; i++) {
                    data(j, i) = 1;
                }
            }
            break;
        // Gradient (from black to white)
        case 3:
            switch (tens) {
            // North direction
            case 0:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(cols - j - 1, i) = double(j * j) / double(cols * cols);
                    }
                }
                break;
            // North-East direction
            case 1:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(cols - j - 1, i) = double(i * j) / double(rows * cols);
                    }
                }
                break;
            // East direction
            case 2:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(j, i) = double(i * i) / double(rows * rows);
                    }
                }
                break;
            // South-East direction
            case 3:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(j, i) = double(i * j) / double(rows * cols);
                    }
                }
                break;
            // South direction
            case 4:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(j, i) = double(j*j) / double(cols * cols);
                    }
                }
                break;
            // South-West direction
            case 5:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(j, rows-i-1) = double(i * j) / double(rows * cols);
                    }
                }
                break;
            // West direction
            case 6:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(j, rows - i - 1) = double(i * i) / double(rows * rows);
                    }
                }
                break;
            // North-West direction
            case 7:
                for (int j = 0; j < cols; j++) {
                    for (int i = 0; i < rows; i++) {
                        data(cols-j-1, rows-i-1) = double(i * j) / double(rows * cols);
                    }
                }
                break;
            default:
                break;
            }
        default:
            break;
    }



    filename = "gradient";/*
    for (int j = 0; j < cols; j++) {
        for (int i = 0; i < rows; i++) {
            data(j,i) = double(i * j) / double(rows * cols);
        }
    }*/
}

Image & Image::change_coeff_ij(const unsigned int i, const unsigned int j, const double val) {
    double value = val;
    if(val < 0.) {
        std::cerr << "Tried inserting value " + std::to_string(val) + " which is negative into an image. Inserting 0 instead" << std::endl;
        value = 0.;
    }
    else if(val > 1.) {
        std::cerr << "Tried inserting value " + std::to_string(val) + " which is above 1 into an image. Inserting 1 instead" << std::endl;
        value = 1.;
    }
    if((0 <= j) && (j < get_cols()) && (0 <= i) && (i < get_rows()))
        data(i,j) = value;
    else
        std::cerr << "Trying to change coeff at coordinates x = " + std::to_string(j) + ", y = " + std::to_string(i) + ", out of bounds. Doing nothing instead" << std::endl;
    return *this;
}  

Image & Image::change_coeff_xy(const unsigned int x, const unsigned int y, const double val) {
    //Matrix mat(*this);
    change_coeff_ij(y, x, val);
    return *this;
}



double Image::max_pixel_value() const {
    double max_value = 0.;
    for(int x = 0; x < data.cols(); x++) {
        for(int y = 0; y < data.rows(); y++) {
            if((*this)(x,y) > max_value)
                max_value = (*this)(x,y);
        }
    }

    // checked it using Eigen method
    //assert(max_value == data.maxCoeff());
    return max_value;
}

double Image::min_pixel_value() const {
    double min_value = 1.;
    for(int x = 0; x < data.cols(); x++) {
        for(int y = 0; y < data.rows(); y++) {
            if((*this)(x,y) < min_value)
                min_value = (*this)(x,y);
        }
    }

    // checked it using Eigen method
    //assert(min_value == data.minCoeff());
    return min_value;
}

Image Image::symmetry_x() const {
    Image img(data.cols(), data.rows(), filename);

    for(int x = 0; x < data.cols(); x++) {
        for(int y = 0; y < data.rows(); y++) {
            img.change_coeff_xy(x, y, (*this)(img.get_cols() - 1 - x, y));
        }
    }

    // checked it using Eigen method
    //assert(img.data == data.rowwise().reverse());
    return img;
}

Image Image::symmetry_y() const {
    Image img(data.cols(), data.rows(), filename);

    for(int x = 0; x < data.cols(); x++) {
        for(int y = 0; y < data.rows(); y++) {
            img.change_coeff_xy(x, y, (*this)(x, img.get_rows() - 1 - y));
        }
    }

    assert(img.data == data.colwise().reverse());
    return img;
}

Image Image::transpose() const {
    Image img(data.rows(), data.cols(), filename);

    for(int x = 0; x < data.cols(); x++) {
        for(int y = 0; y < data.rows(); y++) {
            img.change_coeff_xy(y, x, (*this)(x, y));
        }
    }

    // checked it using Eigen method
    //assert(img.data == data.transpose());
    return img;
}

Image Image::invert_value() const {
    Image img(*this);

    for(int i = 0; i < data.cols(); i ++) {
      for(int j = 0; j < data.rows(); j++) {
        img.change_coeff_xy(i, j, 1. - data(j, i));
      }
    }
    return img;
}
        
Image Image::barycenter() const {
  Image res(data.cols(), data.rows(), 1., "barycenter.png");
  Image inverted = this->invert_value();
  double acc_row = 0;
  double acc_col = 0;
  double sum_of_weights = 0;
  for (int i = 0; i < data.cols(); i++) {
    for (int j = 0; j < data.rows(); j++) {
      acc_col += inverted(i, j) * i;
      acc_row += inverted(i, j) * j;
      sum_of_weights += inverted(i, j);
    }
  }
  int bary_row = std::floor(acc_row / sum_of_weights);
  int bary_col = std::floor(acc_col / sum_of_weights);
  res.change_coeff_xy(bary_col, bary_row, 0.);
  return res;
}

void Image::barycenter(int& row,int& col) const {
  Image inverted = this->invert_value();
  double acc_row = 0;
  double acc_col = 0;
  double sum_of_weights = 0;
  for (int i = 0; i < data.cols(); i++) {
    for (int j = 0; j < data.rows(); j++) {
      acc_col += inverted(i, j) * i;
      acc_row += inverted(i, j) * j;
      sum_of_weights += inverted(i, j);
    }
  }
  int bary_row = std::floor(acc_row / sum_of_weights);
  int bary_col = std::floor(acc_col / sum_of_weights);
  row = bary_row;
  col = bary_col;
}

Image Image::draw_rectangle(unsigned int x, unsigned int y, unsigned int w, unsigned int h, double val) const {
    Image img(*this);
    for (int i = x; i <= x + w; i++)
        for (int j = y; j <= y + h; j++)
            img.change_coeff_xy(i, j, val);
    return img;
}

/*Image Image::change_pixel(unsigned int x, unsigned int y, double val) const {
    Image img(*this);
    img.data(x,y) = val;
    return img;
}*/

Image Image::crop(unsigned int x, unsigned int y, unsigned int w, unsigned int h) const {
    Image img(w, h, "patch_" + std::to_string(x) + "_" + std::to_string(y) + "_" + std::to_string(w) + "_" + std::to_string(h) + ".png");
    img.data = data(Eigen::seqN(x, w), Eigen::seqN(y, h));
    return img;
}

cv::Mat Image::to_opencv() const {
    cv::Mat image;
    cv::eigen2cv(data, image);
    //std::cout << image << std::endl;
    return image;
}

void Image::display() const {
    std::cout << "Displaying image " << filename << ", press any key." << std::endl;
    cv::imshow(filename, to_opencv());
    cv::waitKey(0);
}

void Image::save() const {
  cv::Mat image;
  cv::eigen2cv((data * 255).eval(), image);
  cv::imwrite(filename, image);
}

void Image::save_as(const std::string f) const {
    if (f != "") {
        if (f != filename) {
            cv::Mat image;
            cv::eigen2cv((data * 255).eval(), image);
            cv::imwrite(f, image);
        }
        else
            std::cerr << "Cannot overwrite " << filename << std::endl;
    }
    else
        std::cerr << "Cannot write to empty filename" << std::endl;

}

bool Image::operator<(const Image &i) const {
    return filename < i.filename;
}

Mask Image::to_mask(const bool inside, const double threshold, const int offset) const {
    // Create a Mask of the right size filled with the outside value
    Mask mask(get_cols(), get_rows(), !inside);

    // Iterate over the columns and get the up- and down-most non-white pixels
    for (int x = 0; x < get_cols(); x++) {
        // Up coord
        int up_y = 0;
        bool found_up = false;
        while (!found_up && up_y < get_rows()) {
            if ((*this)(x, up_y) <= threshold) {
                found_up = true;
            }
            up_y++;
        }

        // Down coord
        int down_y = get_rows() - 1;
        bool found_down = false;
        while (!found_down && down_y >= 0) {
            if ((*this)(x, down_y) <= threshold) {
                found_down = true;
            }
            down_y--;
        }   

        // Fill the mask between these two coordinates
        for (int y = up_y + offset; y <= down_y - offset; y++) {
            mask.change_coeff_xy(x, y, inside);
        }     
    }

    // Iterate over the rows and get the left- and right-most non-white pixels
    for (int y = 0; y < get_rows(); y++) {
        // Left coord
        int left_x = 0;
        bool found_left = false;
        while (!found_left && left_x < get_cols()) {
            if ((*this)(left_x, y) <= threshold) {
                found_left = true;
            }
            left_x++;
        }

        // Right coord
        int right_x = get_cols() - 1;
        bool found_right = false;
        while (!found_right && right_x >= 0) {
            if ((*this)(right_x, y) <= threshold) {
                found_right = true;
            }
            right_x--;
        }   

        // Fill the mask between these two coordinates
        for (int x = left_x + offset; x <= right_x - offset; x++) {
            mask.change_coeff_xy(x, y, inside);
        }     
    }

    return mask;

}

Image Image::apply_mask(const Mask &mask, const double fill_value) const {
    // USE ASSERTIONS TO CHECK THAT THE DIMENSIONS OF THE MASK AND IMAGE MATCH
    Image img(*this);

    for (int x = 0; x < get_cols(); x++) {
        for (int y = 0; y < get_rows(); y++) {
            if (mask(x, y))
                img.change_coeff_xy(x, y, fill_value);
        }
    }

    return img;
}

double Image::MSE(const Image &img) const {
    if (get_cols() != img.get_cols() || get_rows() != img.get_rows()) {
        std::cerr << "Warning: Image instances compared with MSE are not the same size, returning -1" << std::endl;
        return -1.;
    }
    else {
        double squared_diffs = 0.;
        for (int x = 0; x < get_cols(); x++) {
            for (int y = 0.; y < get_rows(); y++) {
                squared_diffs += std::pow((*this)(x,y) - img(x,y), 2);
            }
        }

        return squared_diffs / (get_rows() * get_cols());
    }
        
}

double Image::abs_error(const Image &img) const {
    if (get_cols() != img.get_cols() || get_rows() != img.get_rows()) {
        std::cerr << "Warning: Image instances compared with MSE are not the same size, returning -1" << std::endl;
        return -1.;
    }
    else {
        double abs_diffs = 0.;
        for (int x = 0; x < get_cols(); x++) {
            for (int y = 0.; y < get_rows(); y++) {
                abs_diffs += std::abs((*this)(x,y) - img(x,y));
            }
        }

        return abs_diffs / (get_rows() * get_cols());
    } 
}

Image Image::diff(const Image &img) const {
    return Image((data - img.data).cwiseAbs());
}


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                                  Starter 3                               //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////



Image Image::roll(const unsigned int tx, const unsigned int ty) const{
    // Generate temporary buffers
    cv::Mat tmp = to_opencv().clone();

    // Horizontal rolling (y direction)
    if (ty != 0) {
        // Pick zones
        cv::Mat leftside = to_opencv()(cv::Rect(0, 0, ty, data.rows())).clone();
        cv::Mat other = to_opencv()(cv::Rect(ty, 0, data.cols() - ty, data.rows())).clone();

        // Copy images in correct position (left side copied to the very right)
        other.copyTo(tmp(cv::Rect(0, 0, other.cols, other.rows)));
        leftside.copyTo(tmp(cv::Rect(other.cols, 0, leftside.cols, leftside.rows)));
    }

    // Vertical rolling (x direction)
    if (tx != 0) {
        // Pick zones
        cv::Mat top = tmp(cv::Rect(0, 0, data.cols(), tx)).clone();
        cv::Mat otherbis = tmp(cv::Rect(0, tx, data.cols(), data.rows() - tx)).clone();

        // Copy images in correct position (top copied to the very bottom)
        otherbis.copyTo(tmp(cv::Rect(0, 0, otherbis.cols, otherbis.rows)));
        top.copyTo(tmp(cv::Rect(0, otherbis.rows, top.cols, top.rows)));
    }

    // Convert back to Image object
    Image o_img(tmp.rows, tmp.cols, 0.0, filename);
    cv::cv2eigen(tmp, o_img.data);
    return o_img;
}


Image Image::opencv_convolve(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& M) const{
    //Creating Mat to get result of the convo
    cv::Mat convresult;
    //Creating tmp Mat to convert filter to opencv Mat
    cv::Mat tmp;
    cv::eigen2cv(M, tmp);
    //Actual convolution
    cv::filter2D(to_opencv(), convresult, -1, tmp, cv::Point(-1, -1), 0, 4);
    //Putting things back to Image objects
    Image o_conv(data.rows(), data.cols(), 0.0);
    cv::cv2eigen(convresult, o_conv.data);
    return o_conv;
}


Image Image::DFT() const {
    // Initializing output and tmp buffer
    Image o_dft(data.rows(), data.cols(), 0.0, filename);
    cv::Mat tmp_dft;
    // Computing dft
    cv::dft(to_opencv(), tmp_dft, 0);
    // Getting dft back to an Image object
    cv::cv2eigen(tmp_dft, o_dft.data);
    return o_dft;
}


Image Image::IDFT() const {
    // Initializing output and temporary buffer
    Image o_idft(data.rows(), data.cols(), 0.0, filename);
    cv::Mat tmp_idft;
    // Computing dft
    cv::dft(to_opencv(), tmp_idft, cv::DFT_INVERSE + cv::DFT_SCALE);
    // Getting idft back to an Image object
    cv::cv2eigen(tmp_idft, o_idft.data);
    return o_idft;
}


Image Image::fill(const unsigned int rows, const unsigned int cols) const {
    // Initialize output matrix
    Image output(rows - data.rows(), cols - data.cols(), 0.0, filename);
    cv::Mat o_img;
    // Add some padding
    cv::copyMakeBorder(to_opencv(), o_img, 0, rows - data.rows(), 0, cols - data.cols(), cv::BORDER_CONSTANT, cv::Scalar(0));
    // Convert back to Image object
    cv::cv2eigen(o_img, output.data);
    return output;
}


Image Image::DFT_convolve(const Image& filter) const {
    // Initialize output
    Image o_convo(data.rows(), data.cols(), 0.0, filename + "_convolved_with_" + filter.filename);
    // Perfom both DFT
    Image img_dft = this->DFT();
    Image filter_dft = filter.fill(data.rows(), data.cols()).DFT();
    // Perfom product term by term
    cv::Mat prod;
    cv::mulSpectrums(img_dft.to_opencv(), filter_dft.to_opencv(), prod, 0);
    // Convert back to Image object
    cv::cv2eigen(prod, o_convo.data);
    return o_convo.IDFT().roll(filter.data.rows() / 2, filter.data.cols() / 2);
}

//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                             Particular filters                           //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////

Image Image::median_filter(const std::string shape, const int i_r) const {
    //Checking positive dimension
    if (i_r <= 0){
        std::cerr << "*** Image::median_filter *** : i_r must be positive. Original image returned." << std::endl;
        return *this;
    }
    else{
        // Initialization of output image
        Image o_img(get_cols(), get_rows(), 0.0, "Median_filtered");
        unsigned int r = i_r + 1 - (i_r%2);
        if (i_r%2 == 0) std::cerr << "*** Image::median_filter *** : i_r was even. It has been increased by one." << std::endl;

        // Iterating on each pixel of original image
        for (int i = r/2; i < get_cols()-r/2; i++) {
            for (int j = r/2; j < get_rows()-r/2; j++) {
                // Vector to store patch values
                std::vector<double> patch = {};

                // CIRCLE : Exctracting values in a circular (diameter r) neighborhood
                if (shape == "circle"){
                    for (int p_i = 0; p_i < r; p_i++) {
                        for (int p_j = 0; p_j < r; p_j++) {
                            // Checking radius (a pixel is in the circle if its center is a point of the circle)
                            if (std::sqrt(std::pow(p_i+0.5-r/2.0, 2) + std::pow(p_j+0.5-r/2.0, 2)) <= r/2){
                                patch.push_back(data(j-r/2+p_j, i-r/2+p_i));
                            }
                        }
                    }
                    // Sort the vector 
                    std::sort(patch.begin(), patch.end());
                    // Exctracting median (value such that 50% of all values is smaller or equal).
                    o_img.data(j, i) = patch[patch.size()/ 2 + (1- patch.size()%2)];
                }
                // SQUARE : Exctracting values in a square r*r neighborhood
                else{
                    for (int p_i = 0; p_i < r; p_i++) {
                        for (int p_j = 0; p_j < r; p_j++) {
                            patch.push_back(data(j-r/2+p_j, i-r/2+p_i));
                        }
                    }
                    // Sort the vector 
                    std::sort(patch.begin(), patch.end());
                    // Exctracting median (value such that 50% of all values is smaller or equal).
                    o_img.data(j, i) = patch[r*r / 2 + (1- r*r%2)];
                }
            }
        }
        return o_img;
    }
}


Image Image::gaussian_filter(const int i_N, const double mux, const double sx, const double muy, const double sy) const {
    //Check positive dimension
    if (i_N <= 0){
        std::cerr << "*** Image::gaussian_filter *** : i_N must be positive. Original image returned." << std::endl;
        return *this;
    }
    else{
        //Checking for parity of the DIMENSIONS    
        unsigned int N = i_N + 1 - (i_N%2);
        if (i_N%2 == 0) std::cerr << "*** Image::gaussian_filter *** : i_N was even. It has been increased by one." << std::endl;

        // Initialization of the matrix to build the kernel
        Image kernel(N, N, 0.0, "A_gaussian_kernel");

        // Insertion of the gaussian coefficients
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double coef = double( std::exp(-0.5*(i - mux - N / 2) * (i - mux - N / 2) / (sx * sx))*std::exp( - 0.5 * (j - muy - N / 2) * (j - muy - N / 2) / (sy * sy)));
                kernel.change_coeff_xy(i, j, coef);
            }
        }
        // Computing normalization coefficient
        double S = 0.0;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                S += kernel.data(j, i);
            }
        }
        // Applying normalization coeff
        kernel.data = (1.0 / S)*kernel.data;
        return DFT_convolve(kernel);
    }
}


Image Image::boxblur_filter(const int i_N) const {
    //Check positive dimension
    if (i_N <= 0){
        std::cerr << "*** Image::boxblur_filter *** : i_N must be positive. Original image returned." << std::endl;
        return *this;
    }
    else{
        //Checking for parity of the DIMENSIONS    
        unsigned int N = i_N + 1 - (i_N%2);
        if (i_N%2 <= 0) std::cerr << "*** Image::boxblur_filter *** : i_N was even. It has been increased by one." << std::endl;
        // Initialize matrix
        Image kernel(N, N, 1.0, "A_boxblur_kernel");
        // Fill it with corresponding coefficients 
        kernel.data = (1.0 / (N * N)) * kernel.data;
        return DFT_convolve(kernel);
    }
}


Image Image::gradient_x_filter(const std::string type, const bool normalize) const {
    //User asks for help
    if (type == "Help" || type == "HELP" || type == "help"){
        std::cout << "**********************************************************************\n"
                  << "*                  HELP - Image::gradient_x_filter                   *\n"
                  << "**********************************************************************\n"
                  << "* The types taken into account are :                                 *\n"
                  << "*    - 'Sobel' : applies a Sobel gradient filter.                    *\n"
                  << "*    - 'Scharr' : applies a Scharr gradient filter.                  *\n"
                  << "*    - any random string : applies a Prewitt gradient filter.        *\n"
                  << "*                                                                    *\n"
                  << "* If you see this, a Prewitt filter is being applied.                *\n"
                  << "**********************************************************************" << std::endl;
    }
    // Set parameters for kernel (default is Prewitt)
    double c = 1; // center coeff
    double a = 1; // angle coeff
    if (type == "Sobel") {
        a = 1;
        c = 2;
    }
    else if (type == "Scharr") {
        a = 3;
        c = 10;
    }

    // Initialize matrix and norm
    double norm = 1.0;
    Image kernel(3, 3, 0.0, "The_horizontal_"+ type +"_operator");

    // Change first column
    kernel.change_coeff_xy(0, 0, a);
    kernel.change_coeff_xy(0, 1, c);
    kernel.change_coeff_xy(0, 2, a);

    // Change last (third) column
    kernel.change_coeff_xy(2, 0, -a);
    kernel.change_coeff_xy(2, 1, -c);
    kernel.change_coeff_xy(2, 2, -a);

    // Normalize operator
    if (normalize) {
        Image tmp = DFT_convolve(kernel);
        double max = std::abs(tmp.max_pixel_value());
        double min = std::abs(tmp.min_pixel_value());
        if (min < max) norm = max;
        else norm = min;
    }

    // Apply possible normalization
    kernel.data = (1.0 / norm) * kernel.data;

    return DFT_convolve(kernel);
}


Image Image::gradient_y_filter(const std::string type, const bool normalize) const {
    //User asks for help
    if (type == "Help" || type == "HELP" || type == "help"){
        std::cout << "**********************************************************************\n"
                  << "*                  HELP - Image::gradient_y_filter                   *\n"
                  << "**********************************************************************\n"
                  << "* The types taken into account are :                                 *\n"
                  << "*    - 'Sobel' : applies a Sobel gradient filter.                    *\n"
                  << "*    - 'Scharr' : applies a Scharr gradient filter.                  *\n"
                  << "*    - any random string : applies a Prewitt gradient filter.        *\n"
                  << "*                                                                    *\n"
                  << "* If you see this, a Prewitt filter is being applied.                *\n"
                  << "**********************************************************************" << std::endl;
    }
    // Set parameters for kernel (default is Prewitt)
    double c = 1; // center coeff
    double a = 1; // angle coeff
    if (type == "Sobel") {
        a = 1;
        c = 2;
    }
    else if (type == "Scharr") {
        a = 3;
        c = 10;
    }

    // Initialize matrix and norm
    double norm = 1.0;
    Image kernel(3, 3, 0.0, "The_horizontal_" + type + "_operator");

    // Change first column
    kernel.change_coeff_xy(0, 0, a);
    kernel.change_coeff_xy(1, 0, c);
    kernel.change_coeff_xy(2, 0, a);

    // Change last (third) column
    kernel.change_coeff_xy(0, 2, -a);
    kernel.change_coeff_xy(1, 2, -c);
    kernel.change_coeff_xy(2, 2, -a);

    // Normalize operator
    if (normalize) {
        Image tmp = DFT_convolve(kernel);
        double max = std::abs(tmp.max_pixel_value());
        double min = std::abs(tmp.min_pixel_value());
        if (min < max) norm = max;
        else norm = min;
    }

    // Apply possible normalization
    kernel.data = (1.0 / norm) * kernel.data;

    return DFT_convolve(kernel);
}


Image Image::gradient_filter(const std::string type, const bool normalize) const {
    // Compute horizontal and vertical gradient using chosen operators
    Image Sx = gradient_x_filter(type, normalize);
    Image Sy = gradient_y_filter(type, normalize);

    // Initialize output result matrix and norm
    Image o_res(data.cols(), data.rows(), 0.0, filename + "_"+ type +"_operator_processing");

    // Computing the norm of the gradient
    for (int j = 0; j < data.rows(); j++) {
        for (int i = 0; i < data.cols(); i++) {
            if (normalize) o_res.change_coeff_xy(i, j, (1.0/std::sqrt(2))*std::sqrt(Sx.data(j, i) * Sx.data(j, i) + Sy.data(j, i) * Sy.data(j, i)));
            else o_res.change_coeff_xy(i,j, std::sqrt(Sx.data(j, i) * Sx.data(j, i) + Sy.data(j, i) * Sy.data(j, i)));
        }
    }

    return o_res;
}


int factorial(const int n){
    return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}


Image Image::partial_differentiation_boxblur_filter(const int diff_order, const int d_points, const std::string direction) const{

    //User asks for help
    if (direction == "Help" || direction == "HELP" || direction == "help"){
        std::cout << "*******************************************************************************************\n"
                  << "*                  HELP - Image::partial_differentiation_boxblur_filter                   *\n"
                  << "*******************************************************************************************\n"
                  << "* The directions taken into account are :                                                 *\n"
                  << "*    - 'horizontal' : applies a horizontal gradient filter.                               *\n"
                  << "*    - any random string : applies a vertical gradient filter.                            *\n"
                  << "*                                                                                         *\n"
                  << "* If you see this, a vertical gradient filter is being applied.                           *\n"
                  << "*******************************************************************************************" << std::endl;
    }

    //Checking for non positive order
    if (diff_order <= 0 || d_points <= 0){
        std::cerr << "*** Image::partial_differentiation_boxblur_filter *** : Order of differentiation AND approximation must be positive. Original image returned." << std::endl;
        return *this;
    }
    else{
        // VANDERMONDE
        // Size of Vandermonde matrix
        int q = d_points + diff_order - 1 - (1 - diff_order%2);
        if (d_points == 1) q += 1;

        // Initializing storage
        Image vandermonde(q, q, 0.0, "Vandermonde");
        int exponent = 1;
        // Fill the matrix with symetric ints wrt 0 ( example : -1 1 2)
        // Odd case
        if (q%2 == 1){
            for (int i = 0 ; i < q ; i++){
                for (int j = 0 ; j<q ; j++){
                    vandermonde.change_coeff_xy(j,i, std::pow(j-q/2, exponent));
                }
                exponent += 1;
            }
        }
        // Even case
        if (q%2 == 0){
            for (int i = 0 ; i < q ; i++){
                for (int j = 0 ; j < q ; j++){
                    if (j < q/2) vandermonde.change_coeff_xy(j,i, std::pow(j-q/2, exponent));
                    else vandermonde.change_coeff_xy(j,i, std::pow(j-q/2+1, exponent));
                }
                exponent += 1;
            }
        }

        // Create seconde member vector and solve linear system
        Image secondmember(1, q, 0.0, "second_member");
        secondmember.change_coeff_xy(0, diff_order-1, factorial(diff_order));
        // Solve
        Image sol(vandermonde.data.fullPivHouseholderQr().solve(secondmember.data));

        // Turn the scheme into an odd square filter (could be a vector)
        int dimkernel = q + 1 - q%2;
        Image kernel(dimkernel, dimkernel, 0.0, "difference_scheme_filter");
        // Filling kernel
        for (int i = 0 ; i < dimkernel ; i++){
            int shift = 0;
            for (int j = 0 ; j < dimkernel ; j++){

                // HORIZONTAL direction
                if (direction == "horizontal") {
                    // Middle coeff (sum of computed coeffs)
                    if (j==dimkernel/2){
                        kernel.change_coeff_xy(j,i, -sol.data.sum());
                        if (q%2 == 0) shift +=1;
                    }
                    else kernel.change_coeff_xy(j,i, sol.data(j-shift,0));
                }
                // VERTICAL direction
                else {
                    // Middle coeff (sum of computed coeffs)
                    if (j==dimkernel/2){
                        kernel.change_coeff_xy(i,j, -sol.data.sum());
                        if (q%2 == 0) shift +=1;
                    }
                    else kernel.change_coeff_xy(i,j, sol.data(j-shift,0));
                }
            }
        }

        // Normalize operator
        double norm = 0.0;
        Image tmp = DFT_convolve(kernel);
        double max = std::abs(tmp.max_pixel_value());
        double min = std::abs(tmp.min_pixel_value());
        if (min < max) norm = max;
        else norm = min;

        // Apply potential normalization
        kernel.data = (1.0 / norm) * kernel.data;

        return DFT_convolve(kernel);
    }
}


//////////////////////////////////////////////////////////////////////////////
//                                                                          //
//                        Morphological Filters                             //
//                                                                          //
//////////////////////////////////////////////////////////////////////////////


Image Image::binarization(const double t) const
{
    assert(t >= 0.0 && t <= 1.0);
    Image res(*this);
    for (int i = 0; i < this->get_cols(); i++)
    {
        for (int j = 0; j < this->get_rows(); j++)
        {
            if ((*this)(i, j) < t)
            {
                res.change_coeff_xy(i, j, 0.);
            }
            else
            {

                res.change_coeff_xy(i, j, 1.);
            }
        }
    }

    return res;
}







    Image Image::circling_for_filter(const int expansion, const double color ) const 
{
    assert(expansion > 0);
    assert(color>=0.0 && color<= 1.0);
    Image im(this->get_cols()+(2*expansion),this->get_rows()+(2*expansion),color,"");
    for (int i=expansion; i<this->get_rows()+expansion;i++){
        for (int j=expansion; j<this->get_cols()+expansion;j++){

            im.change_coeff_ij(i,j,(*this)[i-expansion][j-expansion]);


        }
    }

    return im;



}


    Image Image::remove_edge_for_filter(const int expansion) const
{
    assert(expansion > 0);
    Image res(this->get_cols()-(2*expansion),this->get_rows()-(2*expansion),1.0,"");
    for (int i=0; i<this->get_rows()-2*expansion;i++){
        for (int j=0; j<this->get_cols()-2*expansion;j++){

            res.change_coeff_ij(i,j,(*this)[i+expansion][j+expansion]);


        }
    }

    return res;



}


    Image Image::dfilter(const int iter, const std::string neighborhood) const
{
    assert(iter>0);
    const int edge = 5;
    Image step = this -> circling_for_filter(edge,1.0);
    Image res = this -> circling_for_filter(edge,1.0);
    Image bin_im = binarization(1.0);
    for (int n = 0; n<iter;n++){
        for (int i = edge; i < (this -> get_rows())+edge;i++){
            for (int j = edge; j< this -> get_cols()+edge;j++){
                
                //if (bin_im[i-edge][j-edge] == 0){
                    if (neighborhood.compare("square")){
                        res.change_coeff_ij(i, j, std::min({step[i-1][j-1], step[i][j-1], step[i+1][j-1],step[i-1][j], step[i][j], step[i+1][j], step[i-1][j+1],step[i][j+1],step[i+1][j+1]}));
                    }
                    else if (neighborhood.compare("cross")){
                        res.change_coeff_ij(i, j, std::min({step[i][j], step[i][j-1], step[i][j+1],step[i-1][j], step[i+1][j]}));
                    }
                    else if (neighborhood.compare("bigcross")){
                        res.change_coeff_ij(i, j, std::min({step[i-1][j-1], step[i][j-1], step[i+1][j-1],step[i-1][j], step[i][j], step[i+1][j], step[i-1][j+1],step[i][j+1],step[i+1][j+1],step[i][j-2],step[i][j+2],step[i-2][j],step[i+2][j]}));

                    }
                    else {
                        std::cerr<<"Please enter a proper patch argument for dfilter";
                    }
                //}
            }   

        }



        step = res;

        
    }

    res = res.remove_edge_for_filter(edge);
    return res;


}    


    Image Image::efilter(const int iter, const std::string neighborhood)const
{
    assert(iter>0);
    const int edge = 5;
    Image step = this -> circling_for_filter(edge,0.0);
    Image res = this -> circling_for_filter(edge,0.0);
    Image bin_im = this -> binarization(1.0);
    for (int n = 0; n<iter;n++){
        for (int i = edge; i < this -> get_rows()+edge;i++){
            for (int j = edge; j< (this -> get_cols())+edge;j++){
                //if (bin_im[i-edge][j-edge]== 0){
                    if (neighborhood.compare("square")){
                        res.change_coeff_ij(i, j, std::max({step[i-1][j-1], step[i][j-1], step[i+1][j-1],step[i-1][j], step[i][j], step[i+1][j], step[i-1][j+1],step[i][j+1],step[i+1][j+1]}));
                    }
                    else if (neighborhood.compare("cross")){
                        res.change_coeff_ij(i, j, std::max({step[i][j], step[i][j-1], step[i][j+1],step[i-1][j], step[i+1][j]}));
                    }
                    else if (neighborhood.compare("bigcross")){
                        res.change_coeff_ij(i, j, std::max({step[i-1][j-1], step[i][j-1], step[i+1][j-1],step[i-1][j], step[i][j], step[i+1][j], step[i-1][j+1],step[i][j+1],step[i+1][j+1],step[i][j-2],step[i][j+2],step[i-2][j],step[i+2][j]}));

                    }
                    else {
                        std::cerr<<"Please enter a proper patch argument for efilter ";
                    }
                //}
            }   

        }
        step = res;


    }
    res=res.remove_edge_for_filter(edge);
    return res;


}    




Image Image::opening_filter(const int iter, const std::string neighborhood) const {
    assert(iter>0);
    Image step(*this);
    Image res(*this);
    for (int n=0;n<iter;n++ ){
        if (neighborhood.compare("square")){
            res = step.efilter(1,"square");
            step = res;
            res = step.dfilter(1,"square");
            step=res;
        }
        else if (neighborhood.compare("cross")){
            res = step.efilter(1,"cross");
            step = res;
            res = step.dfilter(1,"cross");
            step=res;


        }
        else if (neighborhood.compare("bigcross")){
            res = step.efilter(1,"bigcross");
            step = res;
            res = step.dfilter(1,"bigcross");
            step=res;
        
        }
        else {
            std::cerr<<"Please enter a proper patch argument for opening filter ";
        }


    }
    return res;
}



Image Image::closing_filter(const int iter, const std::string neighborhood) const {
    assert(iter>0);
    Image step(*this);
    Image res(*this);
    for (int n=0;n<iter;n++ ){
        if (neighborhood.compare("square")){
            res = step.dfilter(1,"square");
            step = res;
            res = step.efilter(1,"square");
            step=res;
        }
        else if (neighborhood.compare("cross")){
            res = step.dfilter(1,"cross");
            step = res;
            res = step.efilter(1,"cross");
            step=res;


        }
        else if (neighborhood.compare("bigcross")){
            res = step.dfilter(1,"bigcross");
            step = res;
            res = step.efilter(1,"bigcross");
            step=res;
        
        }
        else {
            std::cerr<<"Please enter a proper patch argument for opening filter ";
        }


    }
    return res;
}



Image Image::smoothing_filter(const int iter, const std::string neighborhood) const {
    assert(iter>0);
    Image step(*this);
    Image res(*this);
    for (int n=0;n<iter;n++ ){
        if (neighborhood.compare("square")){
            res = step.opening_filter(1,"square");
            step = res;
            res = step.closing_filter(1,"square");
            step=res;
        }
        else if (neighborhood.compare("cross")){

            res = step.opening_filter(1,"cross");
            step = res;
            res = step.closing_filter(1,"cross");
            step=res;

        }
        else if (neighborhood.compare("bigcross")){

            res = step.opening_filter(1,"bigcross");
            step = res;
            res = step.closing_filter(1,"bigcross");
            step=res;
        }



    }
    return res;
}

Image Image::antismoothing_filter(const int iter, const std::string neighborhood) const {
    assert(iter>0);
    Image step(*this);
    Image res(*this);
    for (int n=0;n<iter;n++ ){
        if (neighborhood.compare("square")){
            res = step.closing_filter(1,"square");
            step = res;
            res = step.opening_filter(1,"square");
            step=res;
        }
        else if (neighborhood.compare("cross")){
            
            res = step.closing_filter(1,"cross");
            step = res;
            res = step.opening_filter(1,"cross");
            step=res;

        }
        else if (neighborhood.compare("bigcross")){

            res = step.closing_filter(1,"bigcross");
            step = res;
            res = step.opening_filter(1,"bigcross");
            step=res;
        }



    }
    return res;
}


Image Image::morphological_gradient_filter() const{

    Image ero((*this).efilter(1,"cross"));
    Image dil((*this).dfilter(1,"cross"));
    Image res((*this));
        for (int i = 0; i < this->get_cols(); i++)
        {
            for (int j = 0; j < this->get_rows(); j++)
            {
                res.change_coeff_xy(i,j,ero(i,j)-dil(i,j));
            }
        }
    return res;


}



float Image::threshold_mean() const
{
    float res = 0;
    for (int i = 0; i < this->get_cols(); i++)
    {
        for (int j = 0; j < this->get_rows(); j++)
        {
            res = res + (*this)(i, j);
        }
    }

    return res / ((this->get_cols()) * (this->get_rows()));
}

float Image::threshold_minmax() const
{

    float step = ((this)->max_pixel_value() - (this)->min_pixel_value()) / 2;
    return (this->min_pixel_value() + step);
}



        
Image Image::convolve(const Matrix& H, char mode, std::string f, float power, int x, int y, 
    float inner_a, float inner_b, float outer_a, float outer_b, char border, float alpha) const {
  // Catch invalid entries.
  if ((border != 'c') and (border != 'e') and (border != 'm')) {
    std::cerr << "Not a valid entry for border, returning original matrix." << std::endl;
    return *this;
  }
  if (mode != 'c' and mode != 'v') { 
    std::cerr << "Invalid entry for kernel alteration, returning original matrix." << std::endl;
    return *this;
  }
  if (inner_a >= outer_a || inner_b >= outer_b) {
    std::cerr << "Inner ellipse should be smaller than outer ellipse, returning original matrix." << std::endl;
    return *this;
  }
  if (H.get_rows() > 2 * data.rows() || H.get_cols() > 2 * data.cols()) {
    std::cerr << "Kernel should not be more than twice as big as original matrix, returning original matrix." << std::endl;
    return *this;
  }
  if (H.get_rows() % 2 == 0 || H.get_cols() % 2 == 0) {
    std::cerr << "Please enter a kernel with an odd number of rows and columns. Returning original matrix." << std::endl;
    return *this;
  }
  // If x and y are not specified, the barycenter will be chosen as center point.
  if (x == -1 && y == -1) { barycenter(x, y); }

  return Image(Matrix<double>::convolve(H, border, mode, x, y, inner_a, inner_b, outer_a, outer_b, power, alpha), f);
}



