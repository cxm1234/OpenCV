//
//  CVProcess.h
//  Study_openCV
//
//  Created by  generic on 2022/10/12.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

NS_ASSUME_NONNULL_BEGIN

@interface CVProcess : NSObject

- (void)handleImg:(UIImage *)img;

/// 生成原图
- (UIImage *)generatorResult;

/// 高斯模糊
- (UIImage *)gaussianBlur;

/// 单通道灰度图
- (UIImage *)cvtColor;

/// canny算法
- (UIImage *)canny;

@end

NS_ASSUME_NONNULL_END
