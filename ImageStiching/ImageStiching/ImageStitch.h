#pragma once

#include <stdio.h>
#include <iostream>
#include "opencv2/opencv.hpp"  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace std;

//�洢ͨ����Ӧ�Ծ������ĸ��ǵ�����
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
	string LeftSrcPath, RightSrcPath;//ͼ��·��
	Mat ImageSrc1, ImageSrc2;//Դͼ��
	Mat ImageGray1, ImageGray2;//�Ҷ�ͼ
	Mat first_match;//
	Mat dst;//ƴ�Ӻ��ͼ��
	//sift����ĵ�
	vector<KeyPoint> keyPoint1, keyPoint2;
	//��Ϊ��˫������̶����������ͼ�����ҵ��㹻����������㵥Ӧ�Ծ�����ôʹ��
	//�洢��һ����Ӧ�Ծ���
	double K[3][3] = { 0.6984010181373395, -0.002258185129717937, 592.0117931126531,
		-0.12906957572453, 0.9948816767358576, -8.026630821766608,
		-0.0004867196664380679, -3.183277072145168e-06, 1 };
	//����ƥ��������������
	vector<vector<DMatch> > matchePoints;
	vector<DMatch> GoodMatchePoints;
	//����ƥ����������
	vector<Point2f> imagePoints1, imagePoints2;
	//��Ӧ�Ծ���
	Mat Homo;

	ConerStruct Corners;//�ĸ��ǵ�����
	Mat imageTransform1;//ֱ��ƴ�Ӻ��ͼ��
public:

	bool ReadImage(void);//����ͼ��
	int FindFeaturesPoints(void);//��������
	void FirstMatch(void);//��ʾ��Ժ��������
	void CalcCorners(void);//�����ĸ�������
	void OptimizeSeam(void);//�Ż�ͼ���ں�
	void ImageRegistration(void);//ͼ����׼���ں�

};
