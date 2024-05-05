/**
 * @file main_1_simulation.cpp
 * @brief Implements methods developped in main 1 simulation
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#include "main_1_simulation.h"


float c1(float i_r){
    float o_coefficient = exp(-i_r);
    return o_coefficient;
}


float c2(float i_r){
    float o_coefficient = 1/(pow((1+i_r),2));
    return o_coefficient;
}


float c3(float i_r){
    float o_coefficient = 1/(log(i_r+1) +1);
    return o_coefficient;
}


float c4(float i_r){
    float o_coefficient = (1-tanh(i_r-3))/2;
    return o_coefficient;
}


float coeff(Pixel i_Pixel_1, Pixel i_Pixel_2, int i_cfun, float i_norm){
    float dist = i_Pixel_1.distance(i_Pixel_2);
    dist=pow(dist,1)/(i_norm/10);
    switch(i_cfun){
        case 1:
            return c1(0.8*dist);
            break;
        case 2:
            return c2(dist);
            break;
        case 3:
            return c3(dist/10);
            break;
        case 4:
            return c4(2*(dist-i_norm/200));
            break;
        default:
            return c1(dist);
    }
}
  

float coeff_anisotropic(Pixel i_Pixel, Pixel i_Pixel_centre, Pixel i_direction, float i_norm,int i_cfun){
    //Get angle between current Pixel and a direction (wrt the central one)
    float cos = i_Pixel.angle_cos(i_Pixel_centre + i_direction, i_Pixel_centre);
    // angle_parameter increasing while angle decrease and always postitive
    float angle_parameter = 0.03*((cos+1)/2);
    //Get distance between considered Pixel and central one
    float dist = i_Pixel.distance(i_Pixel_centre);
    dist=pow(dist,1)/(i_norm/25);
    float o_coeff = c4(angle_parameter*(dist-i_norm/200));
    return o_coeff;
}


Image isotropic(const Image &i_image, Pixel i_centre, int i_cfun){
    //Initializing output image
    Image o_image = Image(i_image);
    //Getting dimensions
    int X = i_image.get_rows();
    int Y = i_image.get_cols();
    //Using Pixel as vector to get diagonal direction
    Pixel diagonal = Pixel(X,Y);
    //Compute diagonal length
    float i_norm = diagonal.length();
    for (int i = 0;i < Y;i++){
        for (int j = 0;j < X;j++){
            Pixel p = Pixel(i,j);
            std::cout << "(" << i  << j << ")" << ": " << coeff(p, i_centre,i_cfun, i_norm) << std::endl;
            if(1){
                double gray_val = 1 - (1 - i_image(i,j))* coeff(p, i_centre, i_cfun, i_norm);
                o_image.change_coeff_xy(i, j, gray_val);
                }
            }
    }
    return o_image;
}


Image anisotropic(const Image &i_image, Pixel i_centre, Pixel i_direction, int i_cfun){
    //Initializing output image
    Image o_image=Image(i_image);  
    //Getting dimensions
    int X = i_image.get_rows();
    int Y = i_image.get_cols();
    //Using Pixel as vector to get diagonal direction
    Pixel diagonal = Pixel(X,Y);
    //Compute diagonal length
    float i_norm = diagonal.length();
    for (int i = 0;i < Y; i++){
        for (int j = 0;j < X; j++){
            Pixel p = Pixel(i,j); 
            double gray_val = 1 - (1 - i_image(i,j)) * coeff_anisotropic(p, i_centre, i_direction, i_norm, i_cfun);
            o_image.change_coeff_xy(i, j, gray_val);
        }
    }
    return o_image;
}

/*
int main(int argc, char** argv)
{
    
    Image i_image("data/clean_finger.png");

    //i_image.display();

    int X = i_image.get_rows();
    int Y = i_image.get_cols();

    Pixel direction = Pixel(1,-2);

    Pixel press_centre = Pixel(1*X/3,2*Y/3);

    //Image o_img = isotropic(i_image, press_centre,4);
    //o_img.save_as("results/isotrpic_4.png");

    Image o_img=anisotropic(i_image, press_centre,direction, 4);
    //o_img.save_as("results/anisotrpic_4_direction.png");
    o_img.display();

    return 0;
}*/
