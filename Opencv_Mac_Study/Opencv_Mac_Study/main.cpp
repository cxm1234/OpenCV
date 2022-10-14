//
//  main.cpp
//  Opencv_Mac_Study
//
//  Created by  generic on 2022/10/14.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, const char * argv[]) {
    // insert code here...
    Mat img = imread("px1.jpg");
    namedWindow("Display", WINDOW_AUTOSIZE);
    imshow("img", img);
    
    waitKey(0);
    return 0;
}
