//
//  main.cpp
//  Opencv_Mac_Study
//
//  Created by  generic on 2022/10/14.
//

#include <iostream>
#include <opencv2/opencv.hpp>

using namespace cv;

void test1();
void test2();

int main(int argc, const char * argv[]) {
    // insert code here...
    test1();
    return 0;
}

void test1() {
    Mat img = imread("px1.jpg");
    namedWindow("Display", WINDOW_AUTOSIZE);
    imshow("img", img);
    
    waitKey(0);
}

void test2() {
    Mat m = Mat::eye(10, 10, CV_32FC1);
    printf("Element is %f\n", m.at<float>(3,3));
}
