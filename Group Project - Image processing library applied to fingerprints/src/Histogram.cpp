#include "Histogram.h"


Histogram::Histogram() {
	bars = std::vector<unsigned int>(256, 0);
	cumsum = std::vector<unsigned int>(256, 0);
}


Histogram::Histogram(const Image& i_img) {
	bars = std::vector<unsigned int>(256, 0);
	for (int i = 0; i < i_img.get_rows(); i++) {
		for (int j = 0; j < i_img.get_cols(); j++) {
			//std::cout << 255*i_img.data.coeff(j,i) << std::endl;
			//bars.at(255*i_img.data.coeff(j,i)) += 1;
			//unsigned int current = bars.at(255 * i_img.data.coeff(j, i));
			bars.at(255.0 * i_img.data(i,j)) += 1;
		}
	}
	//Making the cumulative version
	cumsum = std::vector<unsigned int>(256, 0);
	int k = 0;
	cumsum.at(0) = *(bars.begin());
	for (std::vector<unsigned int>::const_iterator i = bars.begin(); i < bars.end(); i++) {
		if (k == 0) cumsum.at(k) = *i;
		else cumsum.at(k) = *i + cumsum.at(k - 1);
		k += 1;
	}
	std::cout << "Creation of Histogram achieved." << std::endl;
}

unsigned int Histogram::max_class() const{
	unsigned int m = 0;
	for (int i = 0; i <= 255; i++) {
		if (bars.at(i) > m) m = bars.at(i);
	}
	return m;
}


void Histogram::display() const{
	for (std::vector<unsigned int>::const_iterator i = bars.begin(); i < bars.end(); i++) {
		std::cout << *i << std::endl;
	}
}

void Histogram::graph() const {
	int barwidth = 4;
	int w = 256*(barwidth+1)+1;
	int h = w/1.16;
	unsigned int m = max_class();
	Image hist(w,h, 0.5, "");

	for (int i = 0; i <= 255; i++) {
		//std::cout << bars.at(i) << std::endl;
		hist = hist.draw_rectangle( (barwidth + 1)*i, h, barwidth, h-1-h*bars.at(i) / float(m), 1.0);
	}
	hist.display();
}


unsigned int Histogram::operator()(const unsigned int& k){
	return bars.at(k);
}

unsigned int Histogram::operator[](const unsigned int& k) {
	return cumsum.at(k);
}


Image Histogram::hist_equalization(const Image& img) const {
	Image o_img(img.get_cols(), img.get_rows(), 0.0, "Equalized");
	std::cout << "Rows : " << img.get_rows() << " | Cols : " << img.get_cols() << std::endl;
	for (int i = 0; i < img.get_rows(); i++) {
		for (int j = 0; j < img.get_cols(); j++) {
			o_img.change_coeff_xy(j,i, double(cumsum.at(255 * img.data(i,j))) / double(img.data.rows() * img.data.cols())  );
		}
	}
	return o_img;
} 