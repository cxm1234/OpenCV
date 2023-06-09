//
//  CVProcess.m
//  Study_openCV
//
//  Created by  generic on 2022/10/12.
//

#import "CVProcess.h"
#import <opencv2/imgcodecs/ios.h>
#import "itAlpha.hpp"

@implementation CVProcess
{
    ItAlpha *inAlpha;
}

- (void)handleImg:(UIImage *)img {
    
    cv::Mat originMat;
    UIImageToMat(img, originMat);
    
    if (inAlpha) {
        delete inAlpha;
    }
    inAlpha = new ItAlpha(originMat);
}

- (UIImage *)generatorResult {
    cv::Mat output;
    inAlpha->generator(output);
    UIImage *maskImg = MatToUIImage(output);
    return maskImg;
}

- (UIImage *)gaussianBlur {
    cv::Mat output;
    inAlpha->gaussianBlur(output);
    UIImage *gaussianImg = MatToUIImage(output);
    return gaussianImg;
}

- (UIImage *)cvtColor {
    cv::Mat output;
    inAlpha->cvtColor(output);
    UIImage *cvtColor = MatToUIImage(output);
    return  cvtColor;
}

- (UIImage *)canny {
    cv::Mat output;
    inAlpha->canny(output);
    UIImage *canny = MatToUIImage(output);
    return canny;
}


- (UIImage *)clearPaper {
    cv::Mat output;
    inAlpha->clearPaper(output);
    UIImage *clear = MatToUIImage(output);
    return clear;
}

- (UIImage *)sketch {
    cv::Mat output;
    inAlpha->sketch(output);
    UIImage *sketch = MatToUIImage(output);
    return  sketch;
}

- (void)dealloc {
    if (inAlpha) {
        delete inAlpha;
    }
}

@end
