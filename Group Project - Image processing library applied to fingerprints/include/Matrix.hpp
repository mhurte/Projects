/**
 * @file Matrix.hpp
 * @brief 
 * @version 0.1
 * @date 2023-01-04
 * 
 * @copyright Copyright (c) 2023
 * 
 */

#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <string>
#include <eigen3/Eigen/Dense>

float distance(unsigned int x1, unsigned int y1, unsigned int x2, unsigned int y2);

/**
 * @brief Eigen::Matrix wrapper
 * 
 * @tparam T : type of the elements
 */

template <class T>
class Matrix {
    protected:
        /**
         * @brief Wrapped Eigen::Matrix<T, Dynamic, Dynamic>
         */
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> data;

    public:
        /**
         * @brief Construct a new Matrix object
         */
        Matrix() {
        }

        /**
         * @brief Construct a new Matrix object
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         */
        Matrix(const unsigned int cols, const unsigned int rows) : data(rows, cols){
        }

        /**
         * @brief Copy a Matrix object
         * 
         * @param m : matrix to copy
         */
        Matrix(const Matrix &m) {
            data = m.data;
        }

        /**
         * @brief Construct a new Matrix object from an Eigen::Matrix
         * 
         * @param m : Eigen::Matrix
         */
        Matrix(const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> &m) : data(m.cols(), m.rows()) {
            data = m;
        }

        /**
         * @brief Construct a new Matrix object with a default value
         * 
         * @param cols : number of columns
         * @param rows : number of rows
         * @param val : default value
         */
        Matrix(const unsigned int cols, const unsigned int rows, const T val) : data(rows, cols) {
            for (int i = 0; i < cols; i++) {
                for (int j = 0; j < rows; j++) {
                    data(j,i) = val;
                }
            }
        }

        /**
         * @brief Get the width Cols
         * 
         * @return Eigen::Index
         */
        Eigen::Index get_cols() const {
            return data.cols();
        }

        /**
         * @brief Get the height Rows
         * 
         * @return Eigen::Index
         */
        Eigen::Index get_rows() const {
            return data.rows();
        }

        /**
         * @brief Overload operator(x,y) to access elements using functional syntax (COLUMN-MAJOR ORDER)
         * 
         * @param x : column
         * @param y : row
         * @return T
         */
        T operator()(const unsigned int x, const unsigned int y) const {
            if (x > get_cols() || y > get_rows())
                std::cerr << "Cannot access (" << x << ", " << y << ")"
                << ", max indices are " << get_cols() - 1 << " and " << get_rows() - 1 << " respectively" << std::endl;
            
            return data(y, x);
        }
        /**
         * @brief Overload operator* to multiplay matrix with float
         * 
         * @param x : float
         * @return Matrix 
         */

        Matrix operator*(const float x) const {
          Matrix res(*this);
          res.data *= x;
          return res;
        }
        
        /**
         * @brief Overload operator+ to add a float to every element of a matrix
         * 
         * @param x : float
         * @return Matrix 
         */

        Matrix operator+(float x) const {
          Matrix res(*this);
          for(int i = 0; i < data.cols(); i ++) {
            for(int j = 0; j < data.rows(); j++) {
              res.change_coeff_xy(i, j, res(i, j) + x);
            }
          }
          return res;
        }

        /**
         * @brief Overload operator+ to add two matrix objects
         * 
         * @param M : Matrix
         * @return Matrix 
         */

        Matrix operator+(const Matrix& M) const {
          Matrix res(*this);
          if(get_cols() != M.get_cols() || get_rows() != M.get_rows()){
            std::cerr << "Matrices in addition have different sizes, returning left operand" << std::endl;
          }
          else {
            for(int i = 0; i < data.cols(); i ++) {
              for(int j = 0; j < data.rows(); j++) {
                res.change_coeff_xy(i, j, res(i, j) + M(i, j));
              }
            }
          }
          return res;
        }


        /**
         * @brief Overload operator- to subtract two matrix objects
         * 
         * @param M : Matrix
         * @return Matrix 
         */

        Matrix operator-(const Matrix& M) const {
          Matrix res(*this);
          if(get_cols() != M.get_cols() || get_rows() != M.get_rows()){
            std::cerr << "Matrices in substraction have different sizes, returning left operand" << std::endl;
          }
          else {
            for(int i = 0; i < data.cols(); i ++) {
              for(int j = 0; j < data.rows(); j++) {
                res.change_coeff_xy(i, j, res(i, j) - M(i, j));
              }
            }
          }
          return res;
        }
        /**
         * @brief Overload operator[i] to access elements (ROW-MAJOR ORDER). Can be used as matrix[i][j] where i is row, j is column
         * 
         * @param i : row
         * @return T
         */
        Eigen::Matrix<T, 1, Eigen::Dynamic> operator[](const unsigned int i) const {
            if (i > get_rows())
                std::cerr << "Cannot access row " << i << ", max index is " << get_rows() - 1 << std::endl;
            
            return data.row(i);                
        }

        /**
         * @brief Change a given coefficient in the Matrix in (column, row)
         * 
         * @param x : x coord (column)
         * @param y : y coord (row)
         * @param val : value to replace
         * @return Matrix 
         */
        Matrix & change_coeff(const unsigned int x, const unsigned int y, const T val) {
            //Matrix mat(*this);
            std::cerr << "Matrix::change_coeff is DEPRECATED, use Matrix::change_coeff_xy instead" << std::endl;
            data(y,x) = val;
            return *this;
        }

        /**
         * @brief Change a given coefficient in the Matrix in (row, column)
         * 
         * @param i : i coord (row)
         * @param j : j coord (column)
         * @param val : value to replace
         * @return Matrix 
         */
        virtual Matrix & change_coeff_ij(const unsigned int i, const unsigned int j, const T val) {
            //Matrix mat(*this);
            if((0 <= j) && (j < get_cols()) && (0 <= i) && (i < get_rows()))
              data(i,j) = val;
            else
              std::cerr << "Trying to change coeff at coordinates x = " + std::to_string(j) + ", y = " + std::to_string(i) + ", out of bounds. Doing nothing instead" << std::endl;
            return *this;
        }    

        /**
         * @brief Change a given coefficient in the Matrix in (column, row)
         * 
         * @param x : x coord (column)
         * @param y : y coord (row)
         * @param val : value to replace
         * @return Matrix 
         */
        Matrix & change_coeff_xy(const unsigned int x, const unsigned int y, const T val) {
            //Matrix mat(*this);
            change_coeff_ij(y, x, val);
            return *this;
        }

        virtual void display() const {
            std::cout << data << std::endl;
        }

        /**
        * @brief Given the "final" kernel, either return it untouched for constant convo or
        * calculate the evolution of a "unit" kernel into this final kernel
        * depending on radius and angle
        * 
        * @param max_distance : maximal distance of center to edge of picture
        * @param distance_to_center : distance of chosen pixel to center pixel
        * @param mode : manner of change, options are 'c' for constant, 'v' for
        *       variable
        * @param angle : angle of current point to positive x-axis, counter
        *       clockwise
        * @param power : power of energy decrease (e.g. 2 for quadratic)
        * @param inner_a, inner_b, outer_a, outer_b : radii of two ellipses
        *       between which the change will increase (a = height, b = width),
        *       each inner radius has to be smaller (<) than the outer one
        * @return Matrix<T>
        */
        Matrix<T> modify(float max_distance, float distance_to_center, char mode, float angle, 
            float inner_a, float inner_b, float outer_a, float outer_b, float power) const {
          // Infer dimensions of kernel matrix to be applied at the center
          // ("unit" matrix with one 1 in the center).
          Matrix<double> center(data.cols(), data.rows(), 0.);
          center.change_coeff_xy((data.cols() - 1) / 2, (data.rows() - 1) / 2, 1.);
          Matrix<double> res(data.cols(), data.rows(), 0.);
          // Values where alteration to the kernel starts and ends (dependent on
          // radius).
          float begin_change;
          float end_change;
          // Parametrize two ellipses in polar coordinates (scaled with max_distance).
          // If no radii have been set, the inner ellipse reduces to the center point and the outer 
          // is a circle with radius max_distance.
          if (inner_a == -2 || inner_b == -2) {
            begin_change = 0;
          } else {
            begin_change = max_distance * inner_a * inner_b / (std::sqrt(std::pow(inner_a * std::cos(angle), 2) + std::pow(inner_b * std::sin(angle), 2))); 
          }
          if (outer_a == -1 || outer_b == -1) {
            end_change = max_distance;
          } else {
            end_change = max_distance * outer_a * outer_b / (std::sqrt(std::pow(outer_a * std::cos(angle), 2) + std::pow(outer_b * std::sin(angle), 2)));
          }
          switch (mode) {
            case 'c':
                res = (*this);
                break;
            case 'v':
                if (distance_to_center < begin_change) {
                  res = center;
                } else {
                  if (distance_to_center > end_change) {
                    distance_to_center = end_change;
                  }
                  res = center + (*this - center) * std::pow((distance_to_center - begin_change) / (end_change - begin_change), power);
                }
                break;
          }
          return res;
        }

        /**
        * @brief Convolution function with kernel that decreases/blurs/stays constant.
        * 
        * @param H : "final" kernel (the one used on the outside of the picture), 
        *       should be of odd size in both rows and columns and not more than twice as big 
        *       as the matrix (for mirroring)
        * @param x, y : coordinates of chosen center Pixel
        * @param border : how to deal with indices outside of matrix bounds, options are
        *       'c' for constant, 'e' for extended, 'm' for mirrored
        * @param alpha : value around the matrix in case of option "constant"
        * @param mode : kernel alteration -> 'c' for constant, 'v' for variable 
        * @param inner_a, inner_b, outer_a, outer_b : radii of two ellipses
        *       between which the change will increase (a = height, b = width),
        *       each inner radius has to be smaller (<) than the outer one
        * @param filename : new filename
        * 
        * @return Matrix<T>
        */
        Matrix<T> convolve(const Matrix<T>& H, char border, char mode, int x, int y, 
            float inner_a, float inner_b, float outer_a, float outer_b, float power, float alpha) const {
          // Calculate distance of center to furthest corner (for normalization).
          float max_distance = std::max(std::max(distance(0, 0, x, y), distance(data.cols(), 0, x, y)), 
                std::max(distance(0, data.rows(), x, y), distance(data.cols(), data.rows(), x, y)));
          Matrix<T> result(data.cols(), data.rows());
          // Iterate over the original matrix.
          for (int m_c = 0; m_c < data.cols(); m_c++) {
            for (int m_r = 0; m_r < data.rows(); m_r++) {
              float distance_to_center = distance(m_r, m_c, x, y);
              float angle = std::atan2(m_r - x, m_c - y) + M_PI; // Angle between 0 and 2pi.
              Matrix H_modified = H.modify(max_distance, distance_to_center, mode, angle, 
                  inner_a, inner_b, outer_a, outer_b, power);
              T acc(0);
              // Iterate over the kernel.
              for (int h_c = 0; h_c < H.data.cols(); h_c++) {
                for (int h_r = 0; h_r < H.data.rows(); h_r++) {
                  // Calculate index of matrix element to be multiplied.
                  int col = m_c - (h_c - (H.data.cols() - 1) / 2);
                  int row = m_r - (h_r - (H.data.rows() - 1) / 2);
                  // Deal with indices outside of matrix borders.
                  switch (border) {
                    case 'e':
                      if (col < 0) { col = 0; }
                      if (col >= data.cols()) { col = data.cols() - 1; }
                      if (row < 0) { row = 0; }
                      if (row >= data.rows()) { row = data.rows() - 1; }
                      acc += H_modified(h_c, h_r) * (*this)(col, row);
                      break;
                    case 'm':
                      if (col < 0) { col = -col - 1; }
                      if (col >= data.cols()) { col = data.cols() - (col - data.cols()) - 1; }
                      if (row < 0) { row = -row - 1; }
                      if (row >= data.rows()) { row = data.rows() - (row - data.rows()) - 1; }
                      acc += H_modified(h_c, h_r) * (*this)(col, row);
                      break;
                    case 'c':
                      if (col < 0 || col >= data.cols() || row < 0 || row >= data.rows()) { 
                        acc += H_modified(h_c, h_r) * alpha; }
                      else {
                        acc += H_modified(h_c, h_r) * (*this)(col, row); 
                      }
                      break;
                  }
                }
              }
            result = result.change_coeff_xy(m_c, m_r, acc);
            }
          }
          return result;
        }
};


#endif
