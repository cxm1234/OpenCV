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
    
    /// 输出原图
    /// - Parameter output: 输出结果
    void generator(Mat &output);
    
    /// 输出高斯模糊
    /// - Parameter output: 输出结果
    void gaussianBlur(Mat &output);
    
    /// 单通道灰度图
    /// - Parameter output: 输出结果
    void cvtColor(Mat &output);
    
    /// canny
    /// - Parameter output: 输出结果
    void canny(Mat &output);

private:
    Mat origin;
};

#endif /* itAlpha_hpp */
