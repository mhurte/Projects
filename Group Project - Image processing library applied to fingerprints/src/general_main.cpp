#include "Matrix.hpp"
#include "Image.h"
#include "Mask.h"
#include "Dictionary.h"
#include "Transformation.h"
#include "Translation.h"
#include "Rotation.h"
#include "Scaling.h"
#include "Shear.h"
#include "PixelCoordinates.h"
#include "inpainting.h"
#include "transform.h"
#include "local_warp.h"
#include <iostream>
#include <vector>
#include <opencv2/imgproc.hpp>
#include "Histogram.h"
#include "main_1_simulation.h"
#include <fstream>
#include <chrono> //to compute time complexity

int main(int argc, char *argv[])
{
    // Possible first arguments
    std::vector<std::string> parts = {"starter_1", "main_1_simulation", "main_1_restauration", 
                                    "starter_2", "main_2", 
                                    "starter_3", "main_3_simulation", "main_3_restauration", "main_4", 
                                    "starter_5", "main_5", "histo", "cmplx"};
    if (argc < 2) {
        std::cout << "Use at least one argument from " << parts[0];
        for (int i = 1; i < parts.size(); i++) {
            std::cout << ", " << parts[i];
        }
        std::cout << std::endl;

        return 0;
    }

    std::vector<std::string> argList(argv + 1, argv + argc);

    //////////////////////////////////
    //          STARTER 1           //
    //////////////////////////////////
    if (argList[0] == "starter_1") {
        Image img("data/clean_finger.png");
        int test = img.min_pixel_value();
        test = img.max_pixel_value();


        std::cout << "Min pixel value: " << img.min_pixel_value() << ", max pixel value: " << img.max_pixel_value() << std::endl;
        std::cout << "Displaying the original image" << std::endl;
        img.display();
        std::cout << std::endl;

        std::cout << "Drawing a black rectangle from (50,20) of size (30,40)." << std::endl;
        Image rect = img.draw_rectangle(50, 20, 30, 40, 0.);
        std::cout << "Drawing a white rectangle from (150,170) of size (20,50)." << std::endl;
        rect = rect.draw_rectangle(150, 170, 20, 50, 1.);
        std::cout << "Displaying the modified image" << std::endl;
        rect.display();
        std::string rectangles_file = "starter_1_rectangles.png";
        std::cout << ("Saving to " + rectangles_file) << std::endl;
        std::cout << std::endl;
        rect.save_as(rectangles_file);

        std::cout << "Performing a symmetry along the y axis." << std::endl;
        Image sym_y = img.symmetry_y();
        std::cout << "Displaying the modified image" << std::endl;
        sym_y.display();
        std::string sym_y_file = "starter_1_symmetry_y.png";
        std::cout << ("Saving to " + sym_y_file) << std::endl;
        std::cout << std::endl;
        sym_y.save_as(sym_y_file);

        std::cout << "Performing a symmetry along the x axis." << std::endl;
        Image sym_x = img.symmetry_x();
        std::cout << "Displaying the modified image" << std::endl;
        sym_x.display();
        std::string sym_x_file = "starter_1_symmetry_x.png";
        std::cout << ("Saving to " + sym_x_file) << std::endl;
        std::cout << std::endl;
        sym_x.save_as(sym_x_file);

        std::cout << "Performing a transposition." << std::endl;
        Image transp = img.transpose();
        std::cout << "Displaying the modified image" << std::endl;
        transp.display();
        std::string transp_file = "starter_1_transpose.png";
        std::cout << ("Saving to " + transp_file) << std::endl;
        std::cout << std::endl;
        transp.save_as(transp_file);

    }

    //////////////////////////////////
    //       MAIN 1 SIMULATION      //
    //////////////////////////////////
    else if (argList[0] == "main_1_simulation") {
        Image img("data/clean_finger.png");

        Pixel a(1,0);
        Pixel b(-1,0);
        Pixel o(0,0);
        std::cout << a.angle_cos(b,o) << std::endl;
    }

    //////////////////////////////////
    //       MAIN 1 RESTAURATION    //
    //////////////////////////////////
    else if (argList[0] == "main_1_restauration") {
        std::cout << "Possible arguments: \n"
                  << "- rectangle N:int size:int x:int y:int w:int h:int \n"
                  << "- disk N:int size:int x:int y:int r:int \n" 
                  << "- outline" << std::endl;

        std::string shape = (argList.size() >= 2) ? (argList[1]) : "outline";

        // Number and size of the patches
        int N = (argList.size() >= 3) ? std::stoi(argList[2]) : 9000;
        int size = (argList.size() >= 4) ? std::stoi(argList[3]) : 13;

        // Parameters for rectangular or disk masks
        int x, y, w, h, r;
        if (shape == "rectangle") {
            x = (argList.size() >= 5) ? std::stoi(argList[4]) : 80;
            y = (argList.size() >= 6) ? std::stoi(argList[5]) : 120;
            w = (argList.size() >= 7) ? std::stoi(argList[6]) : 60;
            h = (argList.size() >= 8) ? std::stoi(argList[7]) : 60;
        }
        else if (shape == "disk") {
            x = (argList.size() >= 5) ? std::stoi(argList[4]) : 120;
            y = (argList.size() >= 6) ? std::stoi(argList[5]) : 180;
            r = (argList.size() >= 7) ? std::stoi(argList[6]) : 60;
        }

        // Import the clean finger image
        Image img("data/clean_finger.png");

        // Import the weak finger image
        Image weak("data/weak_finger.png");

        // Generating a dictionary of patches
        std::cout << "Generating a dictionary of " << N << " random patches of size " << size << "x" << size << std::endl;
        Dictionary dict(weak, N, size);

        // Command line mask shape selector
        if (shape == "rectangle") {
            std::cout << "Masking a rectangle. Upper left corner is (" << x << "," << y << "), dimensions are (" << w << "," << h << ")" << std::endl;
            // Mask the image using a rectangle, display, save
            Image weak_masked = weak.draw_rectangle(x, y, w, h, 0.);
            weak_masked.display();
            weak_masked.save_as("main_1_restauration_rectangle_mask.png");

            // Inpaint the image, display, save
            std::cout << "Inpainting it" << std::endl;
            Image out = inpaint_rectangle(weak, dict, x, y, w, h);
            out.display();
            out.save_as("main_1_restauration_rectangle_inpainted.png");
        }
        else if (shape == "disk") {
            std::cout << "Masking a ring. Center is (" << x << "," << y << "), inner radius is " << r << std::endl;

            // Generate a disk mask
            Mask disk_mask = Mask(weak.get_cols(), weak.get_rows(), x, y, r);

            // Apply the mask and save
            weak = weak.apply_mask(disk_mask, 0);
            weak.save_as("main_1_restauration_disk_mask.png");

            // Inpaint the ring
            Image out = inpaint_ring(weak, dict, x, y, r);

            // Generate a mask from clean finger image
            //Mask mask = img.to_mask(false, 0.3).fill_holes(false);

            // Apply it, display, save
            //out = out.apply_mask(mask, 1.);
            out.display();
            out.save_as("main_1_restauration_disk_inpainted.png");
        }
        else {
            // Generate a mask from the weak finger image, 
            // fill it so that it's convex, shrink it so that it's inside the fingerprint
            Mask weak_mask = weak.to_mask(false, 0.1).fill_holes(false).shrink(5);

            // Generate a mask from the clean finger image
            Mask mask = img.to_mask(false, 0.3).fill_holes(false);

            // Compute and display the MSE between the masked clean and weak fingerprints
            Image masked_clean = img.apply_mask(mask, 1.);
            std::cout << "MSE between clean and weak similarly masked fingerprints before inpainting: " << masked_clean.MSE(weak.apply_mask(mask)) << std::endl;

     
            
            // Inpaint
            Image inpainted = inpaint_mask(weak, dict, weak_mask);

            // Display and save
            inpainted.display();
            inpainted.save_as("main_1_restauration_outline.png");

            // Apply the mask from the clean finger image, display, save
            Image out = inpainted.apply_mask(mask);
            out.display();
            out.save_as("main_1_restauration_outline_masked.png");

            // Compute the MSE to the inpainted image

            std::cout << "MSE: " << out.MSE(masked_clean) << std::endl;

            /*
            std::ofstream errors_file;
            errors_file.open("main_1_errors.csv", std::ios::app);
            errors_file << "N, size, MSE" << std::endl;

            std::cout << "Running all evaluations" << std::endl;
            for (int N_val = 1000; N_val <= 10000; N_val += 1000) {
                for (int size_val = 5; size_val <= 13; size_val += 4) {
                    Dictionary dict(weak, N_val, size_val);
                    Image inpainted = inpaint_mask(weak, dict, weak_mask);
                    Image out = inpainted.apply_mask(mask);
                    double mse = out.MSE(masked_input);
                    out.save_as("inpainted/" + std::to_string(N_val) + "_" + std::to_string(size_val) + "_" + std::to_string(mse) + ".png");
                    errors_file << N_val << ", " << size_val << ", " << mse << std::endl;
                    std::cout << "N: " << N_val << ", size: " << size_val << ", MSE:" << mse << std::endl;
                }         
            }
            */
            

        }
        
    }

    //////////////////////////////////
    //          STARTER 2           //
    //////////////////////////////////
    else if (argList[0] == "starter_2") {
        std::cout << "Possible arguments: \n- rotation \n- translation \n- scaling \n- shearing \n- forward_then_inverse" 
                  << "\n\tthen arguments for the given transformation"
                  << "\n- best_transformation" << std::endl;

        std::string arg = (argList.size() >= 2) ? argList[1] : "best_transformation";

        // Inport the image
        Image img("data/clean_finger.png");

        // Display it before transformation
        std::cout << "Displaying original image:" << std::endl;
        img.display();

        Image img_out;
        if (arg == "rotation") {
            double theta = (argList.size() >= 3) ? std::stod(argList[2]) : 0.2;
            std::cout << "Rotating by " + std::to_string(theta) + " radians" << std::endl;

            Rotation transfo(theta);

            img_out = interpolate_1st_inverse(img, transfo);
        }
        else if (arg == "translation") {
            double tx = (argList.size() >= 4) ? std::stod(argList[2]) : 40.2;
            double ty = (argList.size() >= 4) ? std::stod(argList[3]) : -30.7;
            std::cout << "Translating by (" + std::to_string(tx) + ", " + std::to_string(ty) + ") pixels" << std::endl;

            Translation transfo(tx, ty);

            img_out = interpolate_1st_inverse(img, transfo);
        }
        else if (arg == "scaling") {
            double sx = (argList.size() >= 4) ? std::stod(argList[2]) : 0.4;
            double sy = (argList.size() >= 4) ? std::stod(argList[3]) : 0.7;
            std::cout << "Scaling by (" + std::to_string(sx) + ", " + std::to_string(sy) + ") times" << std::endl;

            Scaling transfo(sx, sy);

            img_out = interpolate_1st_inverse(img, transfo);
        }
        else if (arg == "shearing") {
            double s = (argList.size() >= 4) ? std::stod(argList[2]) : 0.3;
            bool horizontal = (argList.size() >= 4) ? (argList[3] == "true") : true;
            std::cout << "Shearing by " + std::to_string(s) + " in the " + (horizontal ? "horizontal" : "vertical") + " direction" << std::endl;

            Shear transfo(s, horizontal);

            img_out = interpolate_1st_inverse(img, transfo);
        }
        else if (arg == "forward_then_inverse") {
            std::cout << "Rotating by 0.3 radians" << std::endl;
            std::cout << "then translating by (20, 30) pixels" << std::endl;

            Rotation transfo1(0.3);
            Translation transfo2(20, 30);

            Transformation transfo = transfo2 * transfo1;

            Image forward = interpolate_1st_inverse(img, transfo);

            std::cout << "Displaying forward transformed image:" << std::endl;
            forward.display();

            std::cout << "Performing the inverse transformation" << std::endl;

            img_out = interpolate_1st_inverse(forward, transfo.inverse());


            img_out.diff(img).save_as("starter_2_forward_then_inverse_diff.png");
        }
        else if (arg == "best_transformation") {
            /*
            REPLICATING THE TRANSFORMATION FROM THE SUBJECT
            3 given points:
            120,168 => 109,188
            89, 267 => 164,280
            209,183 => 187,140
            */

            //Eigen::Matrix3d in_points {{120., 89., 209.}, {168., 267., 183.}, {1., 1., 1.}};
            //Eigen::Matrix<double, 2, 3> out_points {{109., 164., 187.}, {188., 280., 140.}};

            // Homogeneous coordinates of 3 points on the original image
            std::vector<Eigen::Vector3d> in_points;
            in_points.push_back(Eigen::Vector3d(120., 168., 1.));
            in_points.push_back(Eigen::Vector3d(89.,  267., 1.));
            in_points.push_back(Eigen::Vector3d(209., 183., 1.));
            PixelCoordinates in_coord(in_points);

           // Homogeneous coordinates of 3 points on the output image
            std::vector<Eigen::Vector3d> out_points;
            out_points.push_back(Eigen::Vector3d(109., 188., 1.));
            out_points.push_back(Eigen::Vector3d(164., 280., 1.));
            out_points.push_back(Eigen::Vector3d(187., 140., 1.));
            PixelCoordinates out_coord(out_points);

            Transformation solution = transformation_between(in_points, out_points);

            img_out = interpolate_1st_inverse(img, Transformation(solution));
        }
        // Display the transformed image
        std::cout << "Displaying transformed image:" << std::endl;
        img_out.display();

        // Save it
        img_out.save_as("stater_2_" + arg + ".png");

        /*
        Image img_out;
        if (arg == "floor") {
            img_out = floor(in, img, out);
        }
        else if (arg == "forward_first_order") {
            img_out = interpolate_1st_forward(in, img, out);
        }
        else if (arg == "inverse_first_order") {
            img_out = interpolate_1st_inverse(img, transfo);
        }
        else if (arg == "forward_then_inverse") {
            img_out = interpolate_1st_inverse(img, transfo);
            std::cout << "Forward transformation" << std::endl;
            img_out.display();
            std::cout << "Inverse transformation" << std::endl;
            img_out = interpolate_1st_inverse(img_out, transfo.inverse());
        }
        else if (arg == "bicubic") {
            img_out = interpolate_bicubic(img, transfo);
        }
        else if (arg == "best_transformation") {
            //REPLICATING THE TRANSFORMATION FROM THE SUBJECT
            //3 given points:
            //120,168 => 109,188
            //89, 267 => 164,280
            //209,183 => 187,140

            //Eigen::Matrix3d in_points {{120., 89., 209.}, {168., 267., 183.}, {1., 1., 1.}};
            //Eigen::Matrix<double, 2, 3> out_points {{109., 164., 187.}, {188., 280., 140.}};

            // Homogeneous coordinates of 3 points on the original image
            std::vector<Eigen::Vector3d> in_points;
            in_points.push_back(Eigen::Vector3d(120., 168., 1.));
            in_points.push_back(Eigen::Vector3d(89.,  267., 1.));
            in_points.push_back(Eigen::Vector3d(209., 183., 1.));
            PixelCoordinates in_coord(in_points);

           // Homogeneous coordinates of 3 points on the output image
            std::vector<Eigen::Vector3d> out_points;
            out_points.push_back(Eigen::Vector3d(109., 188., 1.));
            out_points.push_back(Eigen::Vector3d(164., 280., 1.));
            out_points.push_back(Eigen::Vector3d(187., 140., 1.));
            PixelCoordinates out_coord(out_points);

            Transformation solution = transformation_between(in_points, out_points);

            img_out = interpolate_1st_inverse(img, Transformation(solution));
        }
        img_out.display();
        img_out.save_as("starter_2_transformed_" + arg + ".png");
        */
    }

    //////////////////////////////////
    //            MAIN 2            //
    //////////////////////////////////
    else if (argList[0] == "main_2") {
        Image image("data/clean_finger.png");

        Image img(image.get_cols(), image.get_rows(), 1., "");

        // Very temporary implementation of local warping
        for (int x = 0; x < image.get_cols(); x++) {
            for (int y = 0; y < image.get_rows(); y++) {
                double x_out = x + delta_x(x, y, image.get_cols()/2, 3*image.get_rows()/4, -2000., 0.001, 0.001, 0.001);
                double y_out = y + delta_y(x, y, image.get_cols()/2, 3*image.get_rows()/4, -2000., 0.001, 0.001, 0.001);

                if (x_out > 0 && x_out < img.get_cols() - 1 && y_out > 0 && y_out < img.get_rows() - 1)
                    img.change_coeff_xy(x, y, image(std::floor(x_out), std::floor(y_out)));
            }
        }

        img.display();

        img.save_as("main_2_test_warp.png");
    }

    //////////////////////////////////
    //          STARTER 3           //
    //////////////////////////////////
    else if (argList[0] == "starter_3") {

        //Presentation of this part in the shell
        std::cout << "*********************************************************************************\n"
                  << "*                         WELCOME TO STARTER 3 PROGRAM                          *\n"
                  << "*********************************************************************************\n"
                  << "\n"
                  << "You will see different effects that have been implemented in this part. For exam-\n"
                  << "ple, you will witness gaussian and median filtering, as well as some edge detection.\n"
                  << "\n"
                  << "All that you are going to see next is made from clean_finger.png\n"
                  << "If not, it will be mentionned in the prints in the consol. We recommand that you \n"
                  << "keep the consol in front of you as it will explain what is going on.\n"
                  << "\n"
                  << "Let's start shall we ?" << std::endl;

        //Load the image
        std::cout << "\n*********************************************************************************\n"
                  << "*                                IMAGE LOADING                                  *\n"
                  << "*********************************************************************************" << std::endl;
        Image image("data/clean_finger.png");

        std::cout << "\n*********************************************************************************\n"
                  << "*                                BOXBLUR FILTER                                 *\n"
                  << "*********************************************************************************" << std::endl;

        //Boxblur filter 19x19 (normal size)
        std::cout << "Applying a boxblur filter of size 19x19...";
        Image boxblurimg = image.boxblur_filter(19);
        std::cout << "OK" << std::endl;
        //Display result
        boxblurimg.display();

        //Boxblur filter 12x12 (even size, should be odd)
        std::cout << "\nApplying a boxblur filter of size 12x12..." << std::endl;
        boxblurimg = image.boxblur_filter(12);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        boxblurimg.display();

        //Boxblur filter (-5)x(-5) (should be positive)
        std::cout << "\nApplying a boxblur filter of size (-5)x(-5) ..." << std::endl;
        boxblurimg = image.boxblur_filter(-5);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        boxblurimg.display();

        std::cout << "\n*********************************************************************************\n"
                  << "*                                GAUSSIAN FILTER                                *\n"
                  << "*********************************************************************************" << std::endl;

        //Gaussian filter 5x5 (everything ok)
        std::cout << "\nApplying a gaussian centered & reduced 5x5 filter ..." << std::endl;
        Image gaussimg = image.gaussian_filter(5);
        std::cout << "OK" << std::endl;
        //Display result
        gaussimg.display();

        //Gaussian filter 11x11 not centered
        std::cout << "\nApplying a gaussian filter with mu_x = 9, size is 11x11 ..." << std::endl;
        gaussimg = image.gaussian_filter(11, 9);
        std::cout << "OK" << std::endl;
        //Display result
        gaussimg.display();
        std::cout << ">>> Wow, did you see that shift ?" << std::endl;

        //Gaussian filter 12x12 (even size, should be odd)
        std::cout << "\nApplying a gaussian filter of size 12x12..." << std::endl;
        gaussimg = image.gaussian_filter(12);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        gaussimg.display();

        //Gaussian filter (-5)x(-5) (should be positive)
        std::cout << "\nApplying a gaussian filter of size (-5)x(-5) ..." << std::endl;
        gaussimg = image.gaussian_filter(-5);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        gaussimg.display();

        std::cout << "\n*********************************************************************************\n"
                  << "*                                  MEDIAN FILTER                                *\n"
                  << "*********************************************************************************" << std::endl;

        //Median filter 5x5 (everything ok)
        std::cout << "\nApplying a median 5x5 filter ..." << std::endl;
        Image medimg = image.median_filter("square", 5);
        std::cout << "OK" << std::endl;
        //Display result
        medimg.display();

        //Median filter 12x12 (even size, should be odd)
        std::cout << "\nApplying a median filter of size 12x12..." << std::endl;
        medimg = image.median_filter("square", 12);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        medimg.display();

        //Median filter (-5)x(-5) (should be positive)
        std::cout << "\nApplying a median filter of size (-5)x(-5) ..." << std::endl;
        medimg = image.median_filter("square", -5);
        std::cout << ">>> Oh, it seems there has been a problem but the program handled it all by itself !" << std::endl;
        //Display result
        medimg.display();

        std::cout << "\n*********************************************************************************\n"
                  << "*                                 GRADIENT FILTER                               *\n"
                  << "*********************************************************************************" << std::endl;

        //Horizontal gradient operator (Prewitt, no normalization)
        std::cout << "\nApplying Prewitt horizontal operator, without normalizing ..." << std::endl;
        Image hgradimg = image.gradient_x_filter("kdflkvjslfj", false);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        //Horizontal gradient operator (Prewitt, normalization)
        std::cout << "\nApplying Prewitt horizontal operator, WITH normalizing ..." << std::endl;
        hgradimg = image.gradient_x_filter("kdflkvjslfj", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        //Horizontal gradient operator (Sobel, normalization)
        std::cout << "\nApplying Sobel horizontal operator, with normalization ..." << std::endl;
        hgradimg = image.gradient_x_filter("Sobel", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        //Horizontal gradient operator (Scharr, normalization)
        std::cout << "\nApplying Scharr horizontal operator, with normalization ..." << std::endl;
        hgradimg = image.gradient_x_filter("Scharr", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        std::cout << "\n>>> Note that the same version of these filters exist as vertical ones." << std::endl;

        //Gradient magnitude operator (Prewitt, normalization)
        std::cout << "\nApplying Prewitt operator, with normalization ..." << std::endl;
        hgradimg = image.gradient_filter("aboquyzen", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        //Gradient magnitude operator (Sobel, normalization)
        std::cout << "\nApplying Sobel operator, with normalization ..." << std::endl;
        hgradimg = image.gradient_filter("Sobel", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        //Gradient magnitude operator (Scharr, normalization)
        std::cout << "\nApplying Scharr operator, with normalization ..." << std::endl;
        hgradimg = image.gradient_filter("Scharr", true);
        std::cout << "OK" << std::endl;
        //Display result
        hgradimg.display();

        std::cout << ">>> As it is possible to see, these filters are used for edge detection." << std::endl;

        //Approximation of second derivative (horizontal)
        std::cout << "\nApproximating the second derivative in the horizontal direction at order  2..." << std::endl;
        image.partial_differentiation_boxblur_filter(2,3).display();
        std::cout << "OK" << std::endl;

        std::cout << "\n*********************************************************************************\n"
                  << "*                               STARTER 3 : THE END                             *\n"
                  << "*********************************************************************************" << std::endl;

        //THE END
        std::cout << "Some other methods are available in the starter 3 part, but were not that relevant to show there.\n\n"
            << "Press any key to end this feed..." << std::endl;
    }

    //////////////////////////////////
    //       MAIN 3 SIMULATION      //
    //////////////////////////////////

    else if (argList[0] == "main_3_simulation") {
         // Load images.
         Image clean("data/clean_finger.png");
         Image blurred("data/blurred_finger.png");
         Image inverted = clean.invert_value();

         ////////////// COMPUTE ERRORS /////////////////////
         //std::cout << "\nMSE between clean_finger and blurred_finger: " << blurred.MSE(clean) << std::endl;
         //std::cout << "Absolute error between clean_finger and blurred_finger: " << blurred.abs_error(clean) << std::endl;

         ////////// GET BARYCENTER /////////////////////////
         // Get barycenter of the image and save it in x and y.
         int x, y;
         clean.barycenter(x, y);

         /////////// CREATE KERNELS ///////////////////////
         
         // Create zero kernel for energy decrease.
         Matrix H1(9, 9, 0.);

         // Create some kernels with coefficients adding up to 1 for blurring.
         Matrix H2(9, 9, 1./81); // Box blur.
         Matrix H3(1, 15, 1./15); // Vertical blur (constant coefficients).
         Matrix H7(15, 1, 1./15); // Horizontal blur (constant coefficients).
         Matrix H5(1, 15, 0.); // Vertical directional blur.
         for (int i = 0; i < 15; i++) {
           H5.change_coeff_xy(0, i, (i + 1.) / 120); 
         }
         
         /////////////////// CONVOLUTION WITH RADIUS-DEPENDENT KERNEL //////////////////////////////
         
         // Linear energy decrease.
         Image convo1 = clean.convolve(H1, 'v', "../lin_decr.png", 1);
         convo1.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo1) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo1) << std::endl;
         convo1.display();*/
         
         // Linear energy increase.
         Image convo2 = inverted.convolve(H1, 'v', "../lin_incr.png", 1, x, y);
         convo2 = convo2.invert_value();
         convo2.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo2) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo2) << std::endl;
         convo2.display();*/
       
         // Quadratic energy increase.
         Image convo3 = inverted.convolve(H1, 'v', "../quad_incr.png", 2, x, y);
         convo3 = convo3.invert_value();
         convo3.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo3) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo3) << std::endl;
         convo3.display();*/

         // Increasing box blur.
         Image convo4 = clean.convolve(H2, 'v', "../box_blur_incr.png", 1, x, y);
         convo4.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo4) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo4) << std::endl;
         convo4.display();*/
         
         // Increasing vertical blur (constant kernel coefficients).
         Image convo5 = clean.convolve(H3, 'v', "../vert_blur_incr.png", 1, x, y);
         convo5.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo5) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo5) << std::endl;
         convo5.display();*/
         
         // Increasing horizontal blur (constant kernel coefficients).
         Image convo13 = clean.convolve(H7, 'v', "../hor_blur_incr.png", 1, x, y);
         convo13.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo13) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo13) << std::endl;
         convo13.display();*/
         
         // Combination of quadratic increase (max_distance scaled with 0.6) and vertical blur --> try to recreate blurred finger.
         Image convo14 = inverted.convolve(H1, 'v', "../quad_inc_max_scaled.png", 2, x, y, 0.01, 0.01, 0.6, 0.6);
         convo14 = convo14.invert_value();
         Image convo7 = convo14.convolve(H3, 'v', "../energy_quad_incr_vert_blur.png", 1, x, y);
         convo7.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo7) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo7) << std::endl;
         convo7.display();*/

        
         /////////////////////// CONVOLUTION WITH ELLIPSES /////////////////////
         
         // Increasing vertical blur (constant kernel coefficients).
         Image convo8 = clean.convolve(H3, 'v', "../vert_blur_incr_ell.png", 1, x, y, .4, .3, .6, .5);
         convo8.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo8) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo8) << std::endl;
         convo8.display();*/
         
         // Quadratically increasing energy.
         Image convo9 = inverted.convolve(H1, 'v', "../energy_lin_incr.png", 1, x, y, .1, .3, .6, .35);
         convo9 = convo9.invert_value();
         convo9.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo9) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo9) << std::endl;
         convo9.display();*/

         // Energy increase with p = 0.9 between two ellipses.
         // The center has been moved down slightly since the print fades out more towards the top.
         Image convo10 = inverted.convolve(H1, 'v', "../incr_energy_ell.png", .9, x + 20, y, .25, .25, .7, .35);
         convo10 = convo10.invert_value();
         convo10.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo10) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo10) << std::endl;
         convo10.display();*/

         // Add increasing vertical blur. Need to pass barycenter manually as it has moved during the last convolution.
         Image convo11 = convo10.convolve(H3, 'v', "../incr_energy_vert_blur_ell.png", 1, x, y, .2, .3, .6, .5);
         convo11.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo11) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo11) << std::endl;
         convo11.display();*/
         
         // Alternative: Add increasing up blur. Need to pass barycenter manually as it has moved during the last convolution.
         Image convo12 = convo10.convolve(H5, 'v', "../incr_energy_up_blur_ell.png", 1, x, y, .2, .3, .6, .5);
         convo12.save();
         /*std::cout << "\nMSE between approximation and blurred_finger: " << blurred.MSE(convo12) << std::endl;
         std::cout << "Absolute error between approximation and blurred_finger: " << blurred.abs_error(convo12) << std::endl;
         convo12.display();*/
    

    }

    //////////////////////////////////
    //       MAIN 3 RESTAURATION    //
    //////////////////////////////////
    else if (argList[0] == "main_3_restauration") {

    }



    //////////////////////////////////
    //            MAIN 4            //
    //////////////////////////////////
    else if (argList[0] == "main_4") {
    Image img("data/clean_finger.png");

        float t0 = 0.2;
        float t1 = img.threshold_mean();
        float t2 = img.threshold_minmax();

        img.display();
        cv::waitKey(0);

        Image morf(img.morphological_gradient_filter());
        morf.display();
        cv::waitKey(0);

        Image out1(img);
    
        Image b(img.binarization(t2));
        // b.display();
        // cv::waitKey(0);
    

        Image cl1(img.efilter(1,"bigcross"));
        cl1.display();
        cv::waitKey(0);


        Image cl2(img.dfilter(1,"bigcross"));
        cl2.display();
        cv::waitKey(0);

        Image cl3(img.opening_filter(1,"bigcross"));
        cl3.display();
        cv::waitKey(0);

        Image cl4(img.closing_filter(1,"square"));
        cl4.display();
        cv::waitKey(0);




    


    }

    //////////////////////////////////
    //          STARTER 5           //
    //////////////////////////////////
    else if (argList[0] == "starter_5") {

    }

    //////////////////////////////////
    //            MAIN 5            //
    //////////////////////////////////
    else if (argList[0] == "main_5") {

    }


    //////////////////////////////////
    //           HISTOGRAMS         //
    //////////////////////////////////
    else if (argList[0] == "histo") {
        Eigen::Matrix<double,100, 100> M;
        Image img(argList[1]);
        //Image img(100, 100, 1.0,"");
        img.display();


        Histogram H(img);
        H.graph();

        //H.display();
        Image eq = H.hist_equalization(img);
        eq.display();
        Histogram(eq).graph();
    }

    //////////////////////////////////
    //           COMPLEXITY         //
    //////////////////////////////////
    else if (argList[0] == "cmplx") {

        // Open file to writte data in
        std::ofstream f;
        f.open("convo_compare.txt");
        f << "Dimension\tTimeConvo\tTimeDFTconvo\n";

        // Constant arbitrairy kernel
        Image kernel(3, 3, 0.0, "A_gaussian_kernel");
        Eigen::Matrix<double, 3, 3> eigenkernel = Eigen::Matrix<double, 3, 3>::Zero(3.0,3.0);
        Matrix mkernel(3, 3, 0.0);
        //cv::cv2eigen(kernel.to_opencv(), eigenkernel);

        double T1 = 0.0;
        double T2 = 0.0;

        // iterate on different sizes
        for (int i = 2; i < std::stoi(argList[1])+1; i++) {
            Image img(std::pow(2,i), std::pow(2, i));
            std::cout << "Here" << std::endl;

            // Naive convo
            T1 = 0.0;
            for (int j = 0; j < 10; j++) {
                auto start = std::chrono::high_resolution_clock::now();
                img.convolve(mkernel, 'c');
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> float_ms = end - start;
                T1 += float_ms.count();
            }
            
            // DFT convo
            T2 = 0.0;
            for (int j = 0; j < 10; j++) {
                auto start = std::chrono::high_resolution_clock::now();
                img.DFT_convolve(kernel);
                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double, std::milli> float_ms_2 = end - start;
                T2 += float_ms_2.count();
            }
            

            f << std::pow(2, i) << "\t" << T1/10.0 <<"\t" << T2/10.0 << std::endl;
        }

        f.close();

    }
    return 0;
}
