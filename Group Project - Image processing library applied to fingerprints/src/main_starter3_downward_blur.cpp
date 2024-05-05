/**
 * @file main_fft.cpp
 * @brief 
 * @version 0.1
 * @date 2023-01-10
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "Image.h"
#include "Matrix.hpp"

int main () {
  Image img("../clean_finger.png");
  img.display(); 
  cv::waitKey();
  // Simulating downward blur.
  Matrix H0(1, 15, 1./15);
  Image conv = img.convolve(H0, 'e');
  conv.save_as("downward_blur_0.png");
  conv.display();
  cv::waitKey();
  Matrix H1(1, 15, 0.);
  for (int i = 0; i < 15; i++) { H1.change_coeff(0, i, i / 105.); }
  H1.display();
  Image conv = img.convolve(H1, 'c');
  conv.save_as("downward_blur_1.png");
  conv.display();
  cv::waitKey();
  Matrix H2(1, 15, 0.);
  for (int i = 0; i < 15; i++) { H2.change_coeff(0, 14 - i, i / 105.); }
  H2.display();
  Image conv = img.convolve(H2, 'c');
  conv.save_as("downward_blur_2.png");
  conv.display();
  cv::waitKey();
  return 0;
}
