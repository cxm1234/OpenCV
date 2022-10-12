//
//  itAlpha.cpp
//  Study_openCV
//
//  Created by  generic on 2022/10/12.
//

#include "itAlpha.hpp"

ItAlpha::ItAlpha(Mat originMat) {
    
    origin = originMat;
    
}

void ItAlpha::generator(Mat &output) {
    
    origin.copyTo(output);
}

void ItAlpha::gaussianBlur(Mat &output) {
    GaussianBlur(origin, output, Size(81,81), 0, 0);
    GaussianBlur(output, output, Size(81,81), 0, 0);
}

ItAlpha::~ItAlpha() {
    
}
