//
//  main.cpp
//  Opencv_Mac_Study
//
//  Created by  generic on 2022/10/14.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

void test1();
void test2();
void test3();
void test4();
void test5();
void test6();

int main(int argc, const char * argv[]) {
    test1();
    return 0;
}

void test1() {
    Mat img = imread("px1.jpg");
    imshow("img", img);
    waitKey(0);
}

void test2() {
    Mat m = Mat::eye(10, 10, CV_32FC1);
    printf("Element is %f\n", m.at<float>(3,3));
}

void test3() {
    Mat m = Mat::eye(10, 10, CV_32FC2);
    printf(
           "Element (3,3) is (%f,%f)\n",
           m.at<cv::Vec2f>(3,3)[0],
           m.at<cv::Vec2f>(3,3)[1]
           );
}

void test4() {
    int sz[3] = {4, 4, 4};
    Mat m(3, sz, CV_32FC3);
    randu(m, -1.0f, 1.0f);
    
    
    float max = 0.0f;
    float len2 = 0.0f;
    MatConstIterator it = m.begin<Vec3f>();
    while (it != m.end<Vec3f>()) {
        len2 = (*it)[0]*(*it)[0] + (*it)[1]*(*it)[1] + (*it)[2]*(*it)[2];
        printf("len2 %f \n",len2);
        if (len2 > max) max = len2;
        it++;
    }
}

void test5() {
    
    const int n_mat_size = 5;
    const int n_mat_sz[] = {n_mat_size, n_mat_size, n_mat_size};
    Mat n_mat(3, n_mat_sz, CV_32FC1);
    
    RNG rng;
    rng.fill(n_mat, RNG::UNIFORM, 0.f, 1.f);
    
    const Mat* arrays[] = {&n_mat, 0};
    Mat my_planes[1];
    NAryMatIterator it(arrays, my_planes);
    
    float s = 0.0f;
    int n = 0;
    
    printf("nplanes %zu\n", it.nplanes);
    
    for (int p = 0; p < it.nplanes; p++, ++it) {
        s += sum(it.planes[0])[0];
        n++;
    }
}

void test6() {
    int size[] = {10,10};
    SparseMat sm(2, size, CV_32F);
    for(int i = 0; i < 10; i++) {
        int idx[2];
        idx[0] = size[0] * rand();
        idx[1] = size[1] * rand();
        sm.ref<float>(idx) += 1.0f;
    }
    
    SparseMatConstIterator_<float> it = sm.begin<float>();
    SparseMatConstIterator_<float> it_end = sm.end<float>();
    
    for(; it != it_end; ++it) {
        const SparseMat::Node *node = it.node();
        printf(" (%3d,%3d) %f\n", node->idx[0], node->idx[1], *it);
    }
}




