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

ItAlpha::~ItAlpha() {
    
}
