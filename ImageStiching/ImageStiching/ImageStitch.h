#pragma once

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

//存储通过单应性矩阵后的四个角的坐标
typedef struct
{
	Point2f left_top;
	Point2f left_bottom;
	Point2f right_top;
	Point2f right_bottom;
}ConerStruct;
class ImageStitch
{
public:
	ImageStitch();
	ImageStitch(string RightPath, string LeftPaht);
	~ImageStitch();

public:
	string LeftSrcPath, RightSrcPath;//图像路径
	Mat ImageSrc1, ImageSrc2;//源图像
	Mat ImageGray1, ImageGray2;//灰度图
	Mat first_match;//
	Mat dst;//拼接后的图像
	//sift检测后的点
	vector<KeyPoint> keyPoint1, keyPoint2;
	//因为是双摄像机固定，如果两幅图像不能找到足够的特征点计算单应性矩阵，那么使用
	//存储的一个单应性矩阵。
	double K[3][3] = { 0.6984010181373395, -0.002258185129717937, 592.0117931126531,
		-0.12906957572453, 0.9948816767358576, -8.026630821766608,
		-0.0004867196664380679, -3.183277072145168e-06, 1 };
	//左右匹配后的联合特征点
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;
	//左右匹配后的特征点
	vector<Point2f> imagePoints1, imagePoints2;
	//单应性矩阵
	Mat Homo;

	ConerStruct Corners;//四个角的坐标
	Mat imageTransform1;//直接拼接后的图像
public:

	bool ReadImage(void);//读入图像
	int FindFeaturesPoints(void);//特征点检测
	void FirstMatch(void);//显示配对后的特征点
	void CalcCorners(void);//计算四个角坐标
	void OptimizeSeam(void);//优化图像融合
	void ImageRegistration(void);//图像配准和融合

};
