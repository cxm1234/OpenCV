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

- (UIImage *)generatorResult;

/// 高斯模糊
- (UIImage *)gaussianBlur;

@end

NS_ASSUME_NONNULL_END
