/**
 * @file Image.h
 * @brief 
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef IMAGE_H
#define IMAGE_H

#include <opencv2/opencv.hpp>
#include "Matrix.hpp"
#include "Mask.h"
#include "Pixel.h"
#include <string>
#include <cmath>


class Image : public Matrix<double> {
    friend class Histogram;
    private:
        std::string filename;
        
    public:

        friend int factorial(const int n);

        /**
         * @brief Construct a new Image object
         * 
         */
        Image();

        /**
         * @brief Construct a new Image object from a file
         * 
         * @param f : filename
         */
        Image(std::string f);

        /**
         * @brief Construct a new Image object
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param f : optional filename
         */
        Image(const unsigned int cols, const unsigned int rows, std::string f);

        /**
         * @brief Copy an Image object
         * 
         * @param i : Image to copy
         */
        Image(const Image &i);

        /**
         * @brief Construct a new Image object from a Mask (e.g. for display purposes)
         * 
         * @param mask : boolean mask
         */
        Image (const Mask &mask);

        /**
         * @brief Construct a new Image object with a default value
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param val : default value
         * @param f : optional filename
         */
        Image(const unsigned int cols, const unsigned int rows, const double val, std::string f);

        /**
         * @brief Construct a new Image object from an Eigen::Matrix
         * 
         * @param m : Eigen::Matrix
         * @param f : filename
         */
        Image(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &m, const std::string f);

        /**
         * @brief Construct a new Image object from a Matrix object
         * 
         * @param m : Matrix
         * @param f : filename
         */
        Image(const Matrix<double> &m, const std::string f);

        /**
         * @brief Construct a new Image object from wanted size and color options.
         * 
         * @param cols Number of columns wanted in the final image.
         * @param rows Number of rows wanted in the final image.
         * @param options Parameter to chose whether the image is a gradient, flat, which orientation etc. Default is flat black.
         *      - Units (type of colour scale) :    [0: flat black] [1: flat gray] [2: flat white]  [3: gradient]
         *      - Tens (orientation b->white):      [0: N]  [1: NE] [2: E]  [3: SE] [4: S]  [5: SW] [6: W]  [7:NW]     
         * 
         * @returns Image with corresponding filling.
         */
        Image(const unsigned int cols, const unsigned int rows, const unsigned int options = 000);

        /**
         * @brief Change a given coefficient in the Image in (row, column)
         * 
         * @param i : i coord (row)
         * @param j : j coord (column)
         * @param val : value to replace
         * @return Image 
         */
        Image & change_coeff_ij(const unsigned int i, const unsigned int j, const double val);

        /**
         * @brief Change a given coefficient in the Image in (column, row)
         * 
         * @param x : x coord (column)
         * @param y : y coord (row)
         * @param val : value to replace
         * @return Image 
         */
        Image & change_coeff_xy(const unsigned int x, const unsigned int y, const double val);

        /**
         * @brief Return the maximum intensity of a pixel
         * 
         * @return double 
         */
        double max_pixel_value() const;

        /**
         * @brief Return the minimum intensity of a pixel
         * 
         * @return double 
         */
        double min_pixel_value() const;

        /**
         * @brief Symmetry along the x axis (w.r.t. y axis)
         * 
         * @return Image 
         */
        Image symmetry_x() const;

        /**
         * @brief Symmetry along the y axis (w.r.t. x axis)
         * 
         * @return Image 
         */
        Image symmetry_y() const;

        /**
         * @brief Image transpose
         * 
         * @return Image 
         */
        Image transpose() const;

        /**
         * @brief invert each value x of an image to 1-x
         * 
         * @return Image 
         */
        Image invert_value() const;
        
        /**
         * @brief Calculate barycenter of an image.
         * 
         * @return Image that marks barycenter, kinda pointless but good for
         * checking
         */
        Image barycenter() const; 

        /**
         * @brief Calculate barycenter of an image.
         * 
         * @return nothing, but save row and col of barycenter in the given
         * pointers
         */
         void barycenter(int& row, int& col) const;
        
        /**
         * @brief Draw a rectangle of a given color on the image
         * 
         * @param x : upper left corner x coordinate
         * @param y : upper left corner y coordinate
         * @param w : rectangle width
         * @param h : rectangle height
         * @param val : grayscale [0,1] value
         * @return Image 
         */
        Image draw_rectangle(unsigned int x, unsigned int y, unsigned int w, unsigned int h, double val) const;

        /**
         * @brief Crop a given patch
         * 
         * @param x : upper left corner x coordinate
         * @param y : upper left corner y coordinate
         * @param w : rectangle width
         * @param h : rectangle height
         * @return Image 
         */
        Image crop(unsigned int x, unsigned int y, unsigned int w, unsigned int h) const;

        /**
         * @brief Return the image in OpenCV format
         * 
         * @return cv::Mat 
         */
        cv::Mat to_opencv() const;

        /**
         * @brief Display the image
         * 
         */
        void display() const;

        /**
         * @brief Save the image as a file
         * 
         */
        void save() const;
        /**
         * @brief Save the image as a file (does not allow overwriting the original filename)
         * 
         * @param f 
         */
        void save_as(const std::string f) const;

        /**
         * @brief Comparison operator to insert into std::set<Image>, compares filenames
         * 
         * @param i 
         * @return true
         * @return false 
         */
        bool operator<(const Image &i) const;

        /**
         * @brief Returns a Mask approximating the convex hull of the fingerprint.
         * 
         * @param inside : boolean value on the inside of the fingerprint
         * @param threshold : value above which a pixel is considered to be white
         * @param offset : number of pixels by which the Mask should be reduced on each side
         * @return Mask 
         */
        Mask to_mask( const bool inside = true, const double threshold = 0.3, const int offset = 0) const;

        /**
         * @brief Apply a Mask to the Image by filling the true area of it with fill_value
         * 
         * @param mask : Mask to apply
         * @param fill_value : value to use in the true area of the Mask
         * @return Image 
         */
        Image apply_mask(const Mask &mask, const double fill_value = 1.) const;

        /**
         * @brief Compute the MSE (Mean Square Error to another image)
         * 
         * @param img : image to compare
         * @return double 
         */
        double MSE(const Image &img) const;

        /**
         * @brief Compute the absolute error to another image
         * 
         * @param img : image to compare
         * @return double 
         */
        double abs_error(const Image &img) const;

        /**
         * @brief Compute the absolute value of the difference to another Image pixel by pixel
         * 
         * @param img : Image to compare to
         * @return Image 
         */
        Image diff(const Image &img) const;

        //////////////////////////////////////////////////////////////////////////////
        //                                                                          //
        //                                  Starter 3                               //
        //                                                                          //
        //////////////////////////////////////////////////////////////////////////////

         /**
         * @brief Perfoms a translation from a given vector (in top left corner direction), like if the image was seen as a torus.
         *
         * @param tx The x component of the translation.
         * @param ty The y component of the translation.
         * @returns Translated image by the (tx, ty) vector.
         */
        Image roll(const unsigned int tx, const unsigned int ty) const;

        /**
         * @brief Naive wrapping of cv::convolve.
         *
         * @param M Eigen matrix to convolve with (it is the left member).
         * @returns Convolved image by the given matrix.
         */
        Image opencv_convolve(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& M) const;

        /**
         * @brief Performs the DFT of an image.
         *
         * @returns The DFT of the image (not processed to be displayed).
         */
        Image DFT() const;

        /**
         * @brief Performs the inverse DFT of an image.
         *
         * @returns The inverse DFT of the image (not processed to be displayed).
         */
        Image IDFT() const;

        /**
         * @brief Add padding to an image placed in the top left corner.
         *
         * @param rows Number of rows to add below the image.
         * @param cols Number of columns to add on the right of the image.
         * @returns Padded image, with original one on the top left corner.
         */
        Image fill(const unsigned int rows, const unsigned int cols) const;

        /**
         * @brief Performs a convolution product with the mean of the DFT.
         *
         * @param filter Image object to take as the right member of the convolution product.
         * @returns The convolution of original image with filter.
         */
        Image DFT_convolve(const Image& filter) const;


        //////////////////////////////////////////////////////////////////////////////
        //                                                                          //
        //                             Particular filters                           //
        //                                                                          //
        //////////////////////////////////////////////////////////////////////////////
        
        /**
         * @brief Performs filtering using the median of values in a square centered patch.
         * 
         * This method looks at each pixel. For each of them, it takes the values of the 
         * pixels in a square neigborhood centered on the considered pixel, exactracts the median
         * of the obtained values and place the corresponding value into a new image.
         * 
         * It helps denoising images with adding as few blurring as possible.
         * 
         * @param shape Shape of the patch to apply. "circle" will apply a circular patch, but
         *              any random string will apply a square patch by default.
         * @param r Length (or diameter) of the square (or circular) patch to center on each pixel. If not positive, original image is returned.
         * 
         * @returns Filtered image.
         */
        Image median_filter(const std::string shape = "square", const int i_r = 3) const;


        /**
         * @brief Apply a gaussian filter to the current Image.
         * 
         * 
         * A gaussian kernel can be seen as the projection on the Oxy plane of a 2D gaussian
         * function. The coefficients of the kernel are the values the gaussian function 
         * would have taken in height at corresponding (x,y) couple. 
         * 
         * In the implementation of this method, the center coefficient has index (0,0), top-left
         * corner one has index (-INT(N/2),-INT(N/2)) and top-right one (-INT(N/2),INT(N/2)), for 
         * a centered and reduced gaussian function.
         * 
         * 
         * @param N Size of the wanted filter (final size is NxN). If not positive, original image is returned. If even, increased by one.
         * @param mux Mean of the horizontal gaussian function. Default at mux=0.
         * @param sx Standard variation of the horizontal gaussian function. Default at sx=1.
         * @param muy Mean of the vertical gaussian function. Default at muy=0.
         * @param sy Standard variation of the vertical gaussian function. Default at sy=1.
         * 
         * @returns Blurred Image with the wanted gaussian kernel.
         */
        Image gaussian_filter(const int i_N, const double mux = 0, const double sx = 1, const double muy = 0, const double sy = 1) const;
        

        /**
         * @brief Apply a uniform blurring kernel on current Image.
         * 
         * A boxblurring kernel is a matrix which coefficient are 1/(N*N), so that
         * the global energy of the image is preserved.
         * 
         * @param N Size of the wanted kernel. Return original image if not positive. Is increased by one if positive but even.
         * 
         * @returns Blurred image with the wanted uniform kernel.
         */
        Image boxblur_filter(const int i_N) const;


        /**
         * @brief Apply the horizontal version of a gradient operator.
         * 
         * @param type String to chose between horizontal "Sobel", "Scharr" or "Prewitt" operator. Default is Prewitt (any random string will set parameters to Prewitt).If the 
         *             user uses 'help' or 'Help' or 'HELP', some help is displayed in the console.
         * @param normalize Boolean to normalize the operator. Default is true.
         * 
         * The horizontal Sobel operator is the matrix
         *      1   0   -1
         *      2   0   -2
         *      1   0   -1
         * 
         * The horizontal Scharr operator is the matrix
         *      3   0   -3
         *      10  0  -10
         *      3   0   -3
         * The horizontal Prewitt operator is the matrix
         *      1   0   -1
         *      1   0   -1
         *      1   0   -1
         * 
         * These operators compute the gradient in Ox direction, thanks to the convolution operation.
         * 
         * @returns The image transformed with corresponding horizontal gradient operator.
         */
        Image gradient_x_filter(const std::string type="", const bool normalize = true) const;


        /**
         * @brief Apply the vertical version of a gradient operator.
         * 
         * @param type String to chose between vertical "Sobel", "Scharr" or "Prewitt" operator. Default is Prewitt (any random string will set parameters to Prewitt). If the 
         *             user uses 'help' or 'Help' or 'HELP', some help is displayed in the console.
         * @param normalize Boolean to normalize the operator. Default is true.
         * 
         * The vertical Sobel operator is the matrix
         *      1   2   1
         *      0   0   0
         *      -1  -2  -1
         * 
         * The vertical Scharr operator is the matrix
         *      3   10  3
         *      0   0   0
         *     -3  -10 -3
         * 
         * The vertical Prewitt operator is the matrix 
         *      1   1   1
         *      0   0   0
         *     -1  -1  -1
         *      
         * These operators compute the gradient in Oy direction, thanks to the convolution operation.
         * 
         * @returns The image transformed with the corresponding vertical operator.
         */
        Image gradient_y_filter(const std::string type="", const bool normalize = true) const;


        /**
         * @brief Apply a gradient operator to the current Image.
         * 
         * @param type String to chose between "Sobel", "Scharr" or "Prewitt" operator. Default is Prewitt (any random string will set parameters to Prewitt).
         * @param normalize Boolean to chose whether or not to normalize the operator.
         * 
         * The general idea is to apply a vertical and horizontal corresponding operator,
         * and then to compute the gradient norm as S = sqrt(Sx^2 + Sy^2), where Sx is the image to 
         * which the horizontal version has been applied, and Sy is the image to which the
         * vertical version has been applied.
         * 
         * Then S is just the norm of the gradient (that can be normalized with the right parameter).
         * 
         * @returns The gradient's norm as an image.
         */
        Image gradient_filter(const std::string type="", const bool normalize = true) const;

        /**
         * @brief Computes partial derivative at wanted order of differentiation and precision, with applying a boxblur filter in orthogonal direction.
         * 
         * This method uses symetric schemes to compute partial derivatives, which automatically makes accuracy of order 1 impossible for first derivative. It solves the
         * good Vandermonde linear system to find the coefficients of the scheme and then places them in a square matrix to be apllied as a filter.
         * 
         * To get the final precision, compute d_points + 1 - diff_order in general, or d_points + 2 - diff_order if the diff_order is even.
         * 
         * @param diff_order Order of wanted differentiation. If not positive, returns the original image.
         * @param d_points Number of discretization points. If not positive, returns the original image.
         * @param direction "horizontal" or "vertical", direction in which the partial derivative has to be computed. Default is "horizontal". Any other random string makes vertical one.
         * @return Image in which the corresponding differentiation has been applied. 
         */
        Image partial_differentiation_boxblur_filter(const int diff_order, const int d_points, const std::string direction="horizontal") const;

        //////////////////////////////////////////////////////////////////////////////
        //                                                                          //
        //                    Morphological Filters (Starter 4  && Main 4)          //
        //                                                                          //
        //////////////////////////////////////////////////////////////////////////////
        


        /**
         * @brief circles the image with a number of white pixels
         * 
         * @param expansion : number of pixels to expand on each side
         * @param color : value of the circling pixels
         * 
        */
        Image circling_for_filter(const int expansion, const double color ) const;

        /**
         * @brief removes edges of an image by a certain amount of pixels
         * 
         * @param expansion : number of pixels that were added on each side
         * 
        */
        Image remove_edge_for_filter(const int expansion) const;

        /**
         * @brief applies a dilation filter on a image with a certain neighborhood
         * 
         * @param edge : size of the edge of the image
         * @param iter : number of applications of the filter
         * @param neighborhood : Type of neighborhood we use for the filter "cross" / "bigcross" / "square"
         * 
        */
        Image dfilter(const int iter, const std::string neighborhood = "cross") const;


        /**
         * @brief applies an erosion filter on a image with a certain neighborhood
         * 
         * @param edge : size of the edge of the image
         * @param iter : number of applications of the filter
         * @param neighborhood : Type of neighborhood we use for the filter "cross" / "bigcross" / "square"
         * 
        */
        Image efilter(const int iter, const std::string neighborhood = "cross") const;



        /**
        * @brief returns the binarized version of the image
        * 
        * @param threshold : any value under this number turns the pixel value to 0, above turns it to 1
        * @param iter : number of applications of the filter
        */
        Image binarization(const double t) const;
        

        /**
         * @brief returns the application of an opening filter to the image
         * 
         * 
         * @param iter : number of applications of the opening filter
         * @param neighborhood : Type of neighborhood we use for the filter "cross" / "bigcross" / "square"
        */
        Image opening_filter(const int iter = 1, const std::string neighborhood = "cross") const;


        Image antismoothing_filter(const int iter = 1, const std::string neighborhood = "cross") const;

        /**
         * @brief returns the application of a closing filter to the image
         * 
         * 
         * @param iter : number of applications of the closing filter
         * @param neighborhood : Type of neighborhood we use for the filter "cross" / "bigcross" / "square"
        */
        Image closing_filter(const int iter = 1, const std::string neighborhood = "cross") const;



        /**
         * @brief returns the application of a smoothing filter to the image
         * 
         * @param iter : number of applications of the smoothing filter
         * @param neighborhood : Type of neighborhood we use for the filter "cross" / "bigcross" / "square"
        */
        Image smoothing_filter(const int iter = 1, const std::string neighborhood = "cross") const;




        /**
         * @brief returns the application of a morphological gradient filter to the image
         * 
         * @param iter : number of applications of the smoothing filter
         * 
        */
        Image morphological_gradient_filter() const;




        /**
         * @brief returns the value in between the highest and lowest values of the pixels
         * 
         * 
        */
        float threshold_minmax() const;


        /**
         * @brief returnes the mean value of all the pixels of the image
         * 
         * 
         * 
        */
        float threshold_mean() const;





        /**
        * @brief Calls convolution function on Matrix.
        * 
        * @param H : "final" kernel (the one on the outside of the image)
        *       -> DOUBLE matrix (since T is double in Image)
        *       Should be of odd size in both rows and columns and not more than twice as big 
        *       as the matrix (for mirroring). 
        * @param x, y : coordinates of chosen center Pixel
        * @param border : how to deal with indices outside of matrix bounds, options are
        *       'c' for constant, 'e' for extended, 'm' for mirrored
        * @param alpha : value around the matrix in case of option "constant"
        * @param mode : change of the kernel towards the outside, 'c' for constant, 'v' for variable
        * @param inner_a, inner_b, outer_a, outer_b : radii of two ellipses
        *       between which the change will increase (a = height, b = width),
        *       each inner radius has to be smaller (<) than the outer one
        * @param filename : new filename
        * 
        * @return Image
        */

        Image convolve(const Matrix& H, char mode, std::string f = "convolution.png", float power = 1, int x = -1, int y = -1, 
            float inner_a = -2, float inner_b = -2, float outer_a = -1, float outer_b = -1, char border = 'e', float alpha = 1.) const;
}; 

#endif
