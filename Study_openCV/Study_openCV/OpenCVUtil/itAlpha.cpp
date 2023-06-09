//
//  itAlpha.cpp
//  Study_openCV
//
//  Created by  generic on 2022/10/12.
//

#include "itAlpha.hpp"
#include <iostream>

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

void ItAlpha::cvtColor(Mat &output) {
    cv::cvtColor(origin, output, cv::COLOR_BGR2GRAY);
}

void ItAlpha::canny(Mat &output) {
    cv::Canny(origin, output, 10, 100, 3, true);
}

void ItAlpha::clearPaper(Mat &output) {
    std::vector<cv::Mat> channels;
    cv::split(origin, channels);
    Mat dst;
    double thresh = cv::threshold(channels[2], dst, 210, 255, THRESH_BINARY + THRESH_OTSU);
    int filter_condition = int(thresh * 0.95);
    cv::threshold(channels[2], dst, filter_condition, 255, THRESH_BINARY);
    Mat dst1;
    Mat three_channel = Mat::zeros(dst.rows, dst.cols, CV_8UC4);
    std::vector<cv::Mat> channels2;
    for (int i=0;i<3;i++) {
        if (i == 2) {
            channels2.push_back(dst);
        } else {
            channels2.push_back(Mat::zeros(dst.rows, dst.cols, CV_8UC1));
        }
    }
    cv::merge(channels2, dst1);
//    cv::merge(channels2, three_channel, dst1);
    
    Mat dst2;
    cv::threshold(origin, dst1, 210, 255, THRESH_BINARY);
    Mat dst3;
  
    std::cout << "dst " << dst.channels() << std::endl;
    
//    std::cout << "dst size" << dst.row(0).size << std::endl;
    
    std::cout << "dst1 " << dst1.channels() << std::endl;
    
//    std::cout << "dst1 size" << dst1.row(0).size << std::endl;
//    print(dst);
//
//    print("------");
//
//    print(dst1);
    
    cv::bitwise_or(dst, dst1, dst2);
    output = dst2;
}

void ItAlpha::sketch(Mat &output) {
    Mat dst = Mat::zeros(origin.rows, origin.cols, CV_8UC1);
    cv::cvtColor(origin, dst, COLOR_RGBA2GRAY);
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3,3));
    cv::morphologyEx(dst, dst, MORPH_GRADIENT, kernel, cv::Point(-1,-1), 5);
    cv::threshold(dst, dst, 80, 80, THRESH_TRUNC);
    cv::bitwise_not(dst, dst);
    output = dst;
}

ItAlpha::~ItAlpha() {
    
}
