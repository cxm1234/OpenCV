//
//  main.cpp
//  Opencv_Mac_Study
//
//  Created by  generic on 2022/10/14.
//

#include <iostream>
#include <cstdlib>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

void test1();
void test2();
void test3();
void test4();
void test5();
void test6();
void test7();
void test8();
void test9();
void test10();
void test11();
void test12();
void test13();
void test14();
void test15();
void test16();
void test17();
void test18();
void test19();
void test20();
void test21();
void test22();
void test23();
void test24();
void test25();

int main(int argc, const char * argv[]) {
    test25();
    return 0;
}

void test1() {
    Mat img = imread("px1.jpg");
    imshow("img", img);
    waitKey(0);
}

void test2() {
    Mat M(3, 4, CV_8UC3, Scalar(0,0,255));
    imshow("img", M);
    print(M);
    waitKey(0);
}

void test3() {
    Mat color_image = imread("px1.jpg", IMREAD_COLOR);
    vector<Mat> channels;
    split(color_image, channels);
    imshow("Blue", channels[0]);
    imshow("Green", channels[1]);
    imshow("Red", channels[2]);
    waitKey(0);
}

vector<uchar> getLogLUT(uchar maxValue) {
    double C = 255 / log10(1 + maxValue);
    vector<uchar> LUT(256, 0);
    for(int i = 0; i < 256; ++i)
        LUT[i] = (int)round(C * log10(1+i));
    return LUT;
}

const int BASE = 1.02;
vector<uchar> getExpLUT(uchar maxValue) {
    double C = 255.0 / (pow(BASE, maxValue) - 1);
    vector<uchar> LUT(256, 0);
    for(int i = 0; i < 256; ++i)
        LUT[i] = (int)round(C * (pow(BASE, i) - 1));
    return LUT;
}

void processImage(Mat& I) {
    double maxVal;
    minMaxLoc(I, NULL, &maxVal);
    vector<uchar> LUT = getExpLUT((uchar) maxVal);
    for(int i = 0; i < I.rows; ++i) {
        for (int j = 0; j < I.cols; ++j)
            I.at<uchar>(i,j) = LUT[I.at<uchar>(i,j)];
    }
}

void test4() {
    Mat image = imread("Lenna.png",IMREAD_GRAYSCALE);
    Mat processed_image = image.clone();
    processImage(processed_image);
    imshow("Input image", image);
    imshow("Processed Image", processed_image);
    waitKey();
}

void test5() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat processed_image;
    input_image.convertTo(processed_image, CV_32F);
    processed_image = processed_image + 1;
    log(processed_image, processed_image);
    normalize(processed_image, processed_image, 0, 255, NORM_MINMAX);
    convertScaleAbs(processed_image, processed_image);
    
    imshow("Input image", input_image);
    imshow("Processed Image", processed_image);
    waitKey(0);
}

void test6() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat filtered_image1;
    boxFilter(
              input_image,
              filtered_image1,
              -1,
              Size(3, 3),
              Point(-1, -1),
              true,
              BORDER_REPLICATE
              );
    Mat filtered_image2;
    boxFilter(
              input_image,
              filtered_image2,
              -1,
              Size(7, 7),
              Point(-1, -1),
              true,
              BORDER_REPLICATE
              );
    Mat filtered_image3;
    boxFilter(
              input_image,
              filtered_image3,
              -1,
              Size(11, 11),
              Point(-1, -1),
              true,
              BORDER_REPLICATE
              );
    Mat filtered_image4;
    boxFilter(
              input_image,
              filtered_image4,
              -1,
              Size(15, 15),
              Point(-1, -1),
              true,
              BORDER_CONSTANT
              );
    imshow("Original Image", input_image);
    imshow("Filtered Image1", filtered_image1);
    imshow("Filtered Image2", filtered_image2);
    imshow("Filtered Image3", filtered_image3);
    imshow("Filtered Image4", filtered_image4);
    waitKey(0);
}

void test7() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat filtered_image;
    blur(input_image, filtered_image, Size(15, 15), Point(-1, -1));
    imshow("Original Image", input_image);
    imshow("Filtered Image", filtered_image);
    waitKey(0);
}

void test8() {
    Mat gaussian_kernel = getGaussianKernel(7, 1.0);
    int rows = gaussian_kernel.rows;
    int cols = gaussian_kernel.cols;
    if (gaussian_kernel.isContinuous()) {
        cols = (cols * rows);
        rows = 1;
    }
    
    cout << "Gaussian Kernel...\n";
    for (int row_idx = 0; row_idx < rows; ++row_idx) {
        double * row_ptr = gaussian_kernel.ptr<double>(row_idx);
        for(int col_idx = 0; col_idx < cols; ++col_idx)
            cout << row_ptr[col_idx] << " ";
        cout << "\n";
    }
}

void test9() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    
    Mat filtered_image1;
    GaussianBlur(input_image, filtered_image1, Size(7, 7), 1.0, 1.0, BORDER_REPLICATE);
    
    Mat filtered_image2;
    GaussianBlur(input_image, filtered_image2, Size(7, 7), 2.0, 2.0, BORDER_REPLICATE);
    
    Mat filtered_image3;
    GaussianBlur(input_image, filtered_image3, Size(7, 7), 3.0, 3.0, BORDER_REPLICATE);
    
    Mat filtered_image4;
    GaussianBlur(input_image, filtered_image4, Size(7, 7), 4.0, 4.0, BORDER_REPLICATE);
    
    imshow("Input Image", input_image);
    
    imshow("Filtered Image1", filtered_image1);
    
    imshow("Filtered Image2", filtered_image2);
    
    imshow("Filtered Image3", filtered_image3);
    
    imshow("Filtered Image4", filtered_image4);
    
    waitKey(0);
}

const int KSIZE = 7;
void test10() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat filtered_image;
    Mat kernel = Mat::ones(KSIZE, KSIZE, CV_32F) / (float)(KSIZE * KSIZE);
    filter2D(input_image, filtered_image, -1, kernel, Point(-1, -1), 0, BORDER_REPLICATE);
    
    imshow("Input Image", input_image);
    imshow("Filtered Image", filtered_image);
    waitKey(0);
}

const int KDEVIATIONS = 80;
void test11() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat kernel_X = getGaussianKernel(input_image.cols, KDEVIATIONS);
    Mat kernel_Y = getGaussianKernel(input_image.rows, KDEVIATIONS);
    Mat kernel_X_transpose;
    transpose(kernel_X, kernel_X_transpose);
    Mat kernel = kernel_Y * kernel_X_transpose;
    
    Mat mask, processed_image;
    normalize(kernel, mask, 0, 1, NORM_MINMAX);
    input_image.convertTo(processed_image, CV_64F);
    multiply(mask, processed_image, processed_image);
    convertScaleAbs(processed_image, processed_image);
    imshow("Vignette", processed_image);
    waitKey(0);
}

void test12() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat binary_image(input_image.size(), input_image.type());
    threshold(input_image, binary_image, 120, 255, THRESH_TOZERO_INV);
    imwrite("binary.png", binary_image);
    imshow("Binary.png", binary_image);
    waitKey(0);
}

void test13() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat binary_image(input_image.size(), input_image.type());
    
//    threshold(input_image, binary_image, 120, 255, THRESH_TOZERO_INV);
    adaptiveThreshold(input_image, binary_image, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 25, 0);
    imwrite("binary.png", binary_image);
    imshow("Binary.png", binary_image);
    
    Mat binary_image1(input_image.size(), input_image.type());
    adaptiveThreshold(input_image, binary_image1, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 25, 0);
    imwrite("binary1.png", binary_image1);
    imshow("Binary1.png", binary_image1);
    
    waitKey(0);
}

void test14() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat dilated_image(input_image.size(), input_image.type());
    Mat eroded_image(input_image.size(), input_image.type());
    Mat element = getStructuringElement(MORPH_RECT, Size(5,5), Point(-1, -1));
    dilate(input_image, dilated_image, element, Point(-1, -1), 1);
    erode(input_image, eroded_image, element, Point(-1, -1), 1);
    
    imshow("Input_Image", input_image);
    imshow("Eroded_Image", eroded_image);
    imshow("Dilated_Image", dilated_image);
    waitKey(0);
}

//Mat computeHistogram(Mat input_image) {
//    Mat histogram = Mat::zeros(256, 1, CV_32S);
//    for(int i = 0; i < input_image.rows; ++i) {
//        for(int j = 0; j < input_image.cols; ++j) {
//            int binIdx = (int)input_image.at<uchar>(i,j);
//            histogram.at<int>(binIdx, 0) += 1;
//        }
//    }
//    return histogram;
//}

Mat computeHistogram(Mat input_image) {
    Mat histogram;
    int channels[] = { 0 };
    int histSize[] = { 256 };
    float range[] = { 0, 256 };
    const float* ranges[] = { range };
    calcHist(&input_image, 1, channels, Mat(), histogram, 1, histSize, ranges);
    return histogram;
}

void test15() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat histogram = computeHistogram(input_image);
    cout << "Histogram...\n";
    for(int i = 0; i < histogram.rows; ++i)
        cout << i << " : " << histogram.at<int>(i, 0) << "\n";
}

Mat computeHistogramWithOpenCV(Mat input_image) {
    Mat histogram;
    int channels[] = { 0 };
    int histSize[] = { 256 };
    float range[] = { 0, 256 };
    const float *ranges[] = { range };
    calcHist(&input_image, 1, channels, Mat(), histogram, 1, histSize, ranges);
    return histogram;
}

void plotHistogram(Mat histogram) {
    int plotWidth = 1024, plotHeight = 400;
    int binWidth = (plotWidth / histogram.rows);
    Mat histogramePlot(plotHeight, plotWidth, CV_8UC3, Scalar(0, 0, 0));
    normalize(histogram, histogram, 0, plotHeight, NORM_MINMAX);
    for(int i = 1; i < histogram.rows; ++i) {
        rectangle(
             histogramePlot,
             Point(
                   (binWidth*(i-1)),
                   (plotHeight - cvRound(histogram.at<float>(i-1,0)))),
             Point(
                   binWidth*i,
                   (plotHeight - cvRound(histogram.at<float>(i,0)))),
             CV_RGB(200, 200, 200),
            FILLED
             );
    }
    imshow("Histogram", histogramePlot);
}

void test16() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat histogram = computeHistogramWithOpenCV(input_image);
    plotHistogram(histogram);
    waitKey(0);
}

Scalar getPlotColor(int chIdx) {
    switch (chIdx) {
        case 0:
            return Scalar(255, 0, 0);
            break;
        case 1:
            return Scalar(0, 255, 0);
            break;
        case 2:
            return Scalar(0 , 0, 255);
            break;
            
        default:
            break;
    }
    return Scalar(255, 255, 255);
}

void plotHistogram(Mat histogram, Mat histogramPlot, int chIdx) {
    int plotWidth = histogramPlot.cols;
    int plotHeight = histogramPlot.rows;
    int binWidth = (plotWidth / histogram.rows);
    normalize(histogram, histogram, 0, plotHeight, NORM_MINMAX);
    
    for (int i = 1; i < histogram.rows; ++i) {
        Scalar plotColor = getPlotColor(chIdx);
        line(
             histogramPlot,
             Point((binWidth*(i-1)),
                   (plotHeight - cvRound(histogram.at<float>(i-1, 0)))
                   ),
             Point((binWidth*i),
                   (plotHeight - cvRound(histogram.at<float>(i, 0)))
                   ),
             plotColor,
             2,
             LINE_AA,
             0
             );
        
    }
}

void test17() {
    Mat input_image = imread("Lenna.png");
    vector<Mat> channels;
    split(input_image, channels);
    int plotWidth = 1024, plotHeight = 400;
    Mat histogramPlot(plotHeight, plotWidth, CV_8UC3, Scalar(0, 0, 0));
    for (int chIdx = 0; chIdx < channels.size(); ++chIdx) {
        Mat channel = channels[chIdx];
        Mat histogram = computeHistogram(channel);
        plotHistogram(histogram, histogramPlot, chIdx);
    }
    imshow("Histogram", histogramPlot);
    waitKey(0);
}

void test18() {
    Mat input_image = imread("Lenna.png");
    int rBins = 32, bBins = 32;
    int histSize[] = {rBins, bBins};
    float rRange[] = { 0, 255};
    float bRange[] = { 0, 255};
    const float *ranges[] = {rRange, bRange};
    int channles[] = { 2, 0};
    Mat histogram;
    calcHist(&input_image, 1, channles, Mat(), histogram, 2, histSize, ranges, true, false);
    double maxValue = 0;
    minMaxLoc(histogram, 0, &maxValue, 0, 0);
    
    int scale = 10;
    Mat histImg = Mat::zeros((bBins * scale), (rBins * scale), CV_8UC3);
    for( int r = 0; r < rBins; r++) {
        for(int b = 0; b < bBins; b++) {
            float binVal = histogram.at<float>(r, b);
            int intensity = cvRound(binVal * 255 / maxValue);
            rectangle(histImg, Point(r * scale, b * scale), Point((r + 1) * scale - 1, (b+1) * scale - 1), Scalar::all(intensity), FILLED);
        }
    }
    
    imshow("H-S Histogram", histImg);
    waitKey(0);
}

Mat getHorizontalDeKernel() {
    Mat horizontalDerKernel = Mat::zeros(3, 3, CV_32F);
    horizontalDerKernel.at<float>(0, 0) = -1.0;
    horizontalDerKernel.at<float>(1, 0) = -1.0;
    horizontalDerKernel.at<float>(2, 0) = -1.0;
    
    horizontalDerKernel.at<float>(0, 2) = 1.0;
    horizontalDerKernel.at<float>(1, 2) = 1.0;
    horizontalDerKernel.at<float>(2, 2) = 1.0;
    return (horizontalDerKernel / 3);
}

Mat getVerticalDerKernel() {
    Mat verticalDerKernel = Mat::zeros(3, 3, CV_32F);
    verticalDerKernel.at<float>(0, 0) = -1.0;
    verticalDerKernel.at<float>(0, 1) = -1.0;
    verticalDerKernel.at<float>(0, 2) = -1.0;
    
    verticalDerKernel.at<float>(2, 0) = 1.0;
    verticalDerKernel.at<float>(2, 1) = 1.0;
    verticalDerKernel.at<float>(2, 2) = 1.0;
    
    return (verticalDerKernel / 3);
}

void test19() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat derivative_horizontal, derivative_vertical;
    Mat scaled_derivative_horizontal, scaled_derivative_vertical;
    
    Mat horizontalDerKernel = getHorizontalDeKernel();
    Mat verticalDerKernel = getVerticalDerKernel();
    
    filter2D(input_image, derivative_horizontal, CV_16S, horizontalDerKernel);
    filter2D(input_image, derivative_vertical, CV_16S, verticalDerKernel);
    
    convertScaleAbs(derivative_horizontal, scaled_derivative_horizontal);
    convertScaleAbs(derivative_vertical, scaled_derivative_vertical);
    imshow("Horizontal_Derivative", scaled_derivative_horizontal);
    imshow("Vertical_Derivative", scaled_derivative_vertical);
    waitKey(0);
}

void test20() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat sobel_filtered_horizontal, sobel_filtered_vertical;
    Mat horizontal_der_scaled, vertical_der_scaled;
    
    Sobel(input_image, sobel_filtered_horizontal, CV_16S, 1, 0);
    Sobel(input_image, sobel_filtered_vertical, CV_16S, 0, 1);
    
    convertScaleAbs(sobel_filtered_horizontal, horizontal_der_scaled);
    convertScaleAbs(sobel_filtered_vertical, vertical_der_scaled);
    
    imshow("Horizontal_Dervative", horizontal_der_scaled);
    imshow("Vertical_Dervative", vertical_der_scaled);
    
    waitKey(0);
}

Mat getGradientMagnitude(Mat xGrad, Mat yGrad) {
    CV_Assert((xGrad.rows == yGrad.rows) && (xGrad.cols == yGrad.cols));
    Mat gradient_magnitude(xGrad.rows, xGrad.cols, CV_16S);
    for(int i = 0; i < xGrad.rows; ++i) {
        for(int j = 0; j < xGrad.cols; ++j) {
            gradient_magnitude.at<short>(i ,j) = abs(xGrad.at<short>(i, j)) + abs(yGrad.at<short>(i, j));
        }
    }
    return gradient_magnitude;
}

Mat thresholdGradientMagnitude(Mat gradient_magnitude, int threshold) {
    Mat edges = Mat::zeros(gradient_magnitude.rows, gradient_magnitude.cols, CV_8U);
    for(int i = 0; i < gradient_magnitude.rows; ++i) {
        for(int j = 0; j < gradient_magnitude.cols; ++j) {
            if (gradient_magnitude.at<short>(i, j) >= threshold)
                edges.at<uchar>(i, j) = 255;
        }
    }
    return edges;
}

void test21() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat x_gradient, y_gradient;
    
    Sobel(input_image, x_gradient, CV_16S, 1, 0);
    Sobel(input_image, y_gradient, CV_16S, 0, 1);
    
    Mat gradient_magnitude = getGradientMagnitude(x_gradient, y_gradient);
    Mat edges_output = thresholdGradientMagnitude(gradient_magnitude, 200);
    
    imshow("Edges", edges_output);
    waitKey(0);
    
}

void test22() {
    Mat input_image = imread("Lenna.png");
    cvtColor(input_image, input_image, COLOR_BGR2GRAY);
    
    Mat edges;
    Canny(input_image, edges, 100, 300, 3, false);
    
    imshow("Edge-Detection", edges);
    waitKey(0);
}

Mat input_image, edges;
char* window_name = "Edge-Detection";
int lowThreshold;

void CannyThreshold(int, void*) {
    Canny(input_image, edges, lowThreshold, (lowThreshold * 3), 3);
    imshow(window_name, edges);
}

void test23() {
    Mat input_image = imread("Lenna.png");
    cvtColor(input_image, input_image, COLOR_BGR2GRAY);
    
    edges.create(input_image.size(), input_image.type());
    
    namedWindow(window_name, WINDOW_AUTOSIZE);
    createTrackbar("Min-Threshold:", window_name, &lowThreshold, 100, CannyThreshold);
    CannyThreshold(0, 0);
    
    waitKey(0);
}

void test24() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat output, scaled_output;
    Laplacian(input_image, output, CV_16S, 3);
    convertScaleAbs(output, scaled_output);
    imshow("Laplcian", scaled_output);
    waitKey(0);
}

float getMean(Mat input) {
    int num_elements = (input.rows * input.cols);
    float sum = 0.0;
    for(int i = 0; i < input.rows; ++i) {
        for(int j = 0; j < input.cols; ++j) {
            sum += input.at<float>(i ,j);
        }
    }
    return (sum / num_elements);
}

float getVariance(Mat input) {
    float mean = getMean(input);
    float sum_of_biases = 0.0;
    int num_of_elements = (input.rows * input.cols);
    
    for(int i = 0; i < input.rows; ++i) {
        for(int j = 0; j < input.cols; ++j) {
            float element_value = input.at<float>(i, j);
            sum_of_biases = ((element_value - mean) * (element_value - mean));
        }
    }
    return (sum_of_biases / num_of_elements);
}

void test25() {
    Mat input_image = imread("Lenna.png", IMREAD_GRAYSCALE);
    Mat laplacian_output;
    
    Laplacian(input_image, laplacian_output, CV_32F);
    float std_dev = getVariance(laplacian_output);
    cout << std_dev << "\n";
}
