//
//  itAlpha.hpp
//  Study_openCV
//
//  Created by  generic on 2022/10/12.
//

#ifndef itAlpha_hpp
#define itAlpha_hpp

#include <stdio.h>
#include <opencv2/imgproc.hpp>

using namespace cv;

class ItAlpha {
    
public:
    // 构造和析构
    ItAlpha(Mat originMat);
    ~ItAlpha();
    
    void generator(Mat &output);
    
    void gaussianBlur(Mat &output);

private:
    Mat origin;
};

#endif /* itAlpha_hpp */
