#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "Image.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>


class Histogram {
	private:
		std::vector<unsigned int> bars;
		std::vector<unsigned int> cumsum;
	public:
		Histogram();
		Histogram(const Image& i_img);
		unsigned int max_class() const;

		void display() const;
		void graph() const;
		//bars
		unsigned int operator()(const unsigned int &k);
		//cumsum
		unsigned int operator[](const unsigned int& k);
		Image hist_equalization(const Image& img) const;
};



#endif
