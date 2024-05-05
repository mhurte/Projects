#include "transform.h"

Image nearest_neighbor(const PixelCoordinates &input_coords, const Image &image, const PixelCoordinates &output_coords) {
    // Generate a white Image of the right size
    Image img(image.get_cols(), image.get_rows(), 1., "");

    // Iterate over every pixel coordinates
    for (int i = 0; i < output_coords.get_N(); i++) {
        // Input coordinates
        Eigen::Vector3d vec_in = input_coords[i];
        int x_in = std::floor(vec_in(0));
        int y_in = std::floor(vec_in(1));

        // Output coordinates
        Eigen::Vector3d vec_out = output_coords[i];
        int x_out = std::floor(vec_out(0));
        int y_out = std::floor(vec_out(1));

        //std::cout << "(" << vec_in(0) << "," << vec_in(1) << "," << vec_in(2) << ") => (" << vec_out(0) << "," << vec_in(1) << "," << vec_in(2) << ")" << std::endl;

        // Change the values in the image
        if (x_out >= 0 && y_out >= 0 && x_out <= image.get_cols() - 1 && y_out <= image.get_rows() - 1)
            img.change_coeff_xy(x_out, y_out, image(x_in, y_in));
    }

    return img;
}

bool compare_pair(const std::pair<double, int> &p1, const std::pair<double, int> &p2) {
    return p1.first < p2.first;
}

// Old comparison implementation using forward transformation then getting the closest pixels, VERY SLOW !
/*
Image interpolate_1st_forward(const PixelCoordinates &input_coords, const Image &pixel_values, const PixelCoordinates &output_coords) {
    // Create a white Image of right dimensions
    Image img(pixel_values.get_cols(), pixel_values.get_rows(), 1., "");

    // >>>> CHANGE THIS !!!
    // Replacing a border 2px wide with white 
    Image image(pixel_values);
    for (int x = 0; x < img.get_cols(); x++) {
        image.change_coeff_xy(x, 0, 1.);
        image.change_coeff_xy(x, 1, 1.);
        image.change_coeff_xy(x, image.get_rows() - 1, 1.);
        image.change_coeff_xy(x, image.get_rows() - 2, 1.);
    }

    for (int y = 0; y < img.get_rows(); y++) {
        image.change_coeff_xy(0, y, 1.);
        image.change_coeff_xy(1, y, 1.);
        image.change_coeff_xy(image.get_cols() - 1, y, 1.);
        image.change_coeff_xy(image.get_cols() - 2, y, 1.);
    }

    // Iterate over the input pixel coordinates
    for (int i = 0; i < input_coords.get_N(); i++) {
        // Input coordinates
        Eigen::Vector3d vec = input_coords[i];
        int x = std::floor(vec(0));
        int y = std::floor(vec(1));

        // Vector to store (squared distance, index) pair and sort them by distance
        std::vector<std::pair<double, int>> squared_distances;

        // Iterate over output coordinates and compute squared distances to grid pixels
        for (int j = 0; j < output_coords.get_N(); j++) {
            Eigen::Vector3d vec_out = output_coords[j];
            Eigen::Vector2d diff = (vec_out - vec)(Eigen::seq(0,1));
            squared_distances.push_back(std::pair<double, int>(diff.squaredNorm(), j));    
        }

        // Get the smallest 4 distances and their indices
        std::partial_sort(squared_distances.begin(), squared_distances.begin() + 4, squared_distances.end(), compare_pair);

        // Accumulators
        double gray_val = 0.;
        double inv_distance_sum = 0.;

        // Iteration variables
        //int j = 0;
        //int count = 0;

        // Interpolate over the 4 closest points
        for (int j = 0; j < 4; j++) {
            // If a pixel is almost exactly in the right place, use it (no interpolation)
            if (std::abs(squared_distances[j].first) < 1e-4) {
                Eigen::Vector3d vec_in = input_coords[squared_distances[j].second];
                gray_val = image(std::floor(vec_in(0)), std::floor(vec_in(1)));
                inv_distance_sum = 1.;
            }
            // Else interpolate over the 4 pixel values
            else {
                double inv_distance = 1./std::sqrt(squared_distances[j].first);
                inv_distance_sum += inv_distance;

                Eigen::Vector3d vec_in = input_coords[squared_distances[j].second];
                gray_val += image(std::floor(vec_in(0)), std::floor(vec_in(1))) * inv_distance; 

                //count++;
            }
            //j++;
        }
        //if (j >= 6)
        //    std::cout << "Increase partial sort range" << std::endl;

        gray_val /= inv_distance_sum;

        // Change the value in the output image
        img.change_coeff_xy(x, y, gray_val);
    }

    return img;    
}
*/

Image interpolate_1st_inverse(const PixelCoordinates &input_coords, const Image &pixel_values, const PixelCoordinates &output_coords) {
    // Create a white Image of the right size
    Image img(pixel_values.get_cols(), pixel_values.get_rows(), 1., "");

    // For every inverse transformed pixel
    for (int i = 0; i < input_coords.get_N(); i++) {
        Eigen::Vector3d vec = input_coords[i];

        // Get its coordinates
        double x = vec(0);
        double y = vec(1);
        //std::cout << x << "," << y << std::endl;

        // Check bounds
        if (x > 0 && x < img.get_cols() - 1 && y > 0 && y < img.get_rows() - 1) {
            // Integer coordinates above
            int x1 = std::ceil(x);
            int y1 = std::ceil(y);
            
            // below
            int x0 = x1 - 1;
            int y0 = y1 - 1;

            // Distances
            double delta_x = x - x0;
            double delta_y = y - y0;

            // Pixel values in these 4 integer coordinates
            double f00 = pixel_values(x0, y0);
            double f01 = pixel_values(x0, y1);
            double f10 = pixel_values(x1, y0);
            double f11 = pixel_values(x1, y1);

            // Interpolate the pixel values
            double val = f00 + (f10 - f00) * delta_x + (f01 - f00) * delta_y + (f11 - f10 - f01 + f00) * delta_x * delta_y;

            // Coordinates : where to write in the output pixel grid
            Eigen::Vector3d vec_out = output_coords[i];
            int x_out = std::floor(vec_out(0));
            int y_out = std::floor(vec_out(1));

            // Change the values
            img.change_coeff_xy(x_out, y_out, val);
        }

    }

    return img;
}

Image interpolate_1st_inverse(const Image &image, const Transformation &transfo) {
    // Output coordinates are the pixel grid (integers)
    PixelCoordinates output_coords(image);

    // Input coordinates using the inverse transformation
    PixelCoordinates input_coords = transfo.inverse() * output_coords;

   return interpolate_1st_inverse(input_coords, image, output_coords);
}

Image interpolate_bicubic(const Image &image, const Transformation &transfo) {
        // Create a white Image of the right size
    Image img(image.get_cols(), image.get_rows(), 1., "");

    // Output coordinates are the pixel grid (integers)
    PixelCoordinates output_coords(image);

    // Input coordinates using the inverse transformation
    PixelCoordinates input_coords = transfo.inverse() * output_coords;

    // First order derivation operators
    // Define centered first order FD scheme convolution kernels
    Eigen::Matrix3d dx = Eigen::Matrix3d::Zero();
    dx(1, 2) =  1;
    dx(1, 0) = -1;
    Matrix<double> Dx(dx);

    Eigen::Matrix3d dy = Eigen::Matrix3d::Zero();
    dy(2, 1) =  1;
    dy(0, 1) = -1;
    Matrix<double> Dy(dy);

    // Compute first order derivatives of the Image using convolution
    Image derivative_x = image.convolve(Dx, 'c', "derivative_x.png");
    Image derivative_y = image.convolve(Dy, 'c', "derivative_y.png");
    //Image derivative_x = image.gradient_x_filter();
    //Image derivative_y = image.gradient_y_filter();

    // Compute crossed derivative of the Image using convolution
    Image derivative_xy = derivative_x.convolve(Dy, 'c', "derivative_xy.png");
    //Image derivative_xy = derivative_x.gradient_y_filter();

    //derivative_x.display();
    //derivative_y.display();
    //derivative_xy.display();

    // For every transformed pixel
    for (int i = 0; i < input_coords.get_N(); i++) {
        Eigen::Vector3d vec = input_coords[i];
        double x = vec(0);
        double y = vec(1);
        //std::cout << x << "," << y << std::endl;

        // Check bounds
        if (x > 0 && x < image.get_cols() - 1 && y > 0 && y < image.get_rows() - 1) {
            // Integer coordinates above
            int x1 = std::ceil(x);
            int y1 = std::ceil(y);
            
            // below
            int x0 = x1 - 1;
            int y0 = y1 - 1;

            // Distances to them
            double w = x1 - x;
            double h = y1 - y;

            // Pixel values in these 4 integer coordinates
            double f00 = image(x0, y0);
            double f01 = image(x0, y1);
            double f10 = image(x1, y0);
            double f11 = image(x1, y1);

            // df/dx values
            double fx00 = derivative_x(x0, y0);
            double fx01 = derivative_x(x0, y1);
            double fx10 = derivative_x(x1, y0);
            double fx11 = derivative_x(x1, y1);

            // df/dy values
            double fy00 = derivative_y(x0, y0);
            double fy01 = derivative_y(x0, y1);
            double fy10 = derivative_y(x1, y0);
            double fy11 = derivative_y(x1, y1);

            // d²f/dxdy values
            double fxy00 = derivative_xy(x0, y0);
            double fxy01 = derivative_xy(x0, y1);
            double fxy10 = derivative_xy(x1, y0);
            double fxy11 = derivative_xy(x1, y1);            

            /*
            // Build a matrix of function values and derivatives
            Eigen::Matrix4d values {{f00,  f01,  fy00,  fy01},
                                    {f10,  f11,  fy10,  fy11},
                                    {fx00, fx01, fxy00, fxy01},
                                    {fx10, fx11, fxy10, fxy11}};

            // Matrices for bicubic coefficients computation
            Eigen::Matrix4d M1 {{1,  0,  0,  0},
                                {0,  0,  1,  0},
                                {-3, 3, -2, -1},
                                {2, -2,  1,  1}};

            Eigen::Matrix4d M2 {{1,  0, -3,  2},
                                {0,  0,  3, -2},
                                {0,  1, -2,  1},
                                {0,  0, -1,  1}};

            Eigen::Matrix4d bicubic_coefficients = M1 * values * M2;
            */

            Eigen::MatrixXd M_inv {{1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
                                   {0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0},
                                   {-3,3,0,0,-2,-1,0,0,0,0,0,0,0,0,0,0},
                                   {2,-2,0,0,1,1,0,0,0,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0},
                                   {0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0},
                                   {0,0,0,0,0,0,0,0,-3,3,0,0,-2,-1,0,0},
                                   {0,0,0,0,0,0,0,0,2,-2,0,0,1,1,0,0},
                                   {-3,0,3,0,0,0,0,0,-2,0,-1,0,0,0,0,0},
                                   {0,0,0,0,-3,0,3,0,0,0,0,0,-2,0,-1,0},
                                   {9,-9,-9,9,6,3,-6,-3,6,-6,3,-3,4,2,2,1},
                                   {-6,6,6,-6,-3,-3,3,3,-4,4,-2,2,-2,-2,-1,-1},
                                   {2,0,-2,0,0,0,0,0,1,0,1,0,0,0,0,0},
                                   {0,0,0,0,2,0,-2,0,0,0,0,0,1,0,1,0},
                                   {-6,6,6,-6,-4,-2,4,2,-3,3,-3,3,-2,-1,-2,-1},
                                   {4,-4,-4,4,2,2,-2,-2,2,-2,2,-2,1,1,1,1}};

            Eigen::VectorXd beta {{f00, f10, f01, f11, fx00, fx10, fx01, fx11, fy00, fy10, fy01, fy11, fxy00, fxy01, fxy10, fxy11}};

            Eigen::VectorXd alpha = M_inv * beta;


            /*
            // Row vector of x powers
            Eigen::RowVector4d X {{1., x, std::pow(x, 2), std::pow(x, 3)}};
            // Column vector of y powers
            Eigen::Vector4d Y {{1., y, std::pow(y, 2), std::pow(y, 3)}};
             

            // Interpolate using the bicubic coefficients
            double val = X * bicubic_coefficients * Y;
            */

            double val = 0.;

            for (int m = 0; m < 4; m++) {
                for (int n = 0; n < 4; n++) {
                    val += alpha(n,m) * std::pow(1 - w, m) * std::pow(1 - h, n);
                }
            }


            // Coordinates : where to write in the pixel grid
            Eigen::Vector3d vec_out = output_coords[i];
            int x_out = std::floor(vec_out(0));
            int y_out = std::floor(vec_out(1));

            // Change the values
            img.change_coeff_xy(x_out, y_out, val);
        }

    }

    return img;
}

Transformation transformation_between(const PixelCoordinates &in, const PixelCoordinates &out) {
    if (in.get_N() != 3 || out.get_N() != 3) {
        std::cerr << "Computing an affine transformation only requires 3 control points, returning identity matrix" << std::endl;
        return Transformation(Eigen::Matrix3d::Identity());
    }
    else {
        // Homogeneous coordinates of 3 points on the original image
        Eigen::Matrix3d in_points {{in[0][0], in[1][0], in[2][0]}, {in[0][1], in[1][1], in[2][1]}, {1., 1., 1.}};
        
        // 2D coordinates of these points on the output image
        Eigen::Matrix<double, 2, 3> out_points {{out[0][0], out[1][0], out[2][0]}, {out[0][1], out[1][1], out[2][1]}};

        // Affine transformation between them
        Eigen::Matrix<double, 3, 2> sol = in_points.transpose().inverse() * out_points.transpose();

        // As a 3x3 homogeneous transformation
        Eigen::Matrix<double, 3, 3> solution;
        solution << sol.transpose(), Eigen::Matrix<double, 1, 3>(0., 0., 1.);

        std::cout << "Best transformation between these points is:\n" << solution << std::endl;

        return Transformation(solution);
    }



}