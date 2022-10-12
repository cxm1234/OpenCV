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
    UIImageToMat(img, originMat, true);
    
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

- (void)dealloc {
    if (inAlpha) {
        delete inAlpha;
    }
}

@end
