#include "ImageStitch.h"


ImageStitch::ImageStitch()
{
	

}
/********************************************************
���������RightPath��ͼ��·����LeftPath��ͼ��·��
���������
����˵�������캯������ͼ��·��
*********************************************************/
ImageStitch::ImageStitch(string RightPath, string LeftPath)
{
	LeftSrcPath = LeftPath;
	RightSrcPath = RightPath;
}

ImageStitch::~ImageStitch()
{
}
/********************************************************
���������
���������
����˵�����Ƿ���ȷ����ͼ��
*********************************************************/
bool ImageStitch::ReadImage(void)
{
	ImageSrc1 = imread(RightSrcPath, 1);
	ImageSrc2 = imread(LeftSrcPath, 1);
	if (ImageSrc1.empty()||ImageSrc2.empty())
	{
		cout << "the images is not existing" << endl;
		return false;
	}
	return true;
}
/********************************************************
���������
���������������Եĸ���
����˵����sift�㷨��ȡ�ص������������
*********************************************************/
int ImageStitch::FindFeaturesPoints(void)
{
	//תΪ�Ҷ�ͼ
	cvtColor(ImageSrc1, ImageGray1, CV_RGB2GRAY);
	cvtColor(ImageSrc2, ImageGray2, CV_RGB2GRAY);
	//��ȡ����ͼȫ����������
	Ptr<Feature2D> Detector = xfeatures2d::SIFT::create(2000);
	Mat imageDesc1, imageDesc2;
	
	Detector->detectAndCompute(ImageSrc1, imageDesc1, keyPoint1, imageDesc1);
	Detector->detectAndCompute(ImageSrc2, imageDesc2, keyPoint2, imageDesc2);

	//knn���ڽ�ƥ���
	FlannBasedMatcher matcher;
	//�Ƚ�����ƥ����ѵ��--û�з���
	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();
	matcher.knnMatch(imageDesc2, matchePoints, 2);//�ҳ���ӦtrainDescritors������ƥ��㣬����������
	cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,��ȡ����ƥ���
	for (int i = 0; i < matchePoints.size(); i++)
	{ 
		//ratio��ֵ����Ϊ0.4��ȡ�߾���ƥ���
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)//��������ĵ�ľ���С�ڵڶ�����0.4
		{
			GoodMatchePoints.push_back(matchePoints[i][0]);
		}
	}
	cout << "last match points: " << GoodMatchePoints.size() << endl;
	for (int i = 0; i<GoodMatchePoints.size(); i++)
	{
		imagePoints2.push_back(keyPoint2[GoodMatchePoints[i].queryIdx].pt);
		imagePoints1.push_back(keyPoint1[GoodMatchePoints[i].trainIdx].pt);
	}
	return GoodMatchePoints.size();
}
/********************************************************
���������
���������
����˵������ʾƥ����
*********************************************************/
void ImageStitch::FirstMatch(void)
{
	
	drawMatches(ImageSrc2, keyPoint2, ImageSrc1, keyPoint1, GoodMatchePoints, first_match);
	
}
/********************************************************
���������
���������
����˵����������ͼ����ĸ��ǵ�ͨ����Ӧ�Ծ���Homoת����
		  ��ͼ������ꡣ��void ImageStitch::ImageRegistration(void)����
�ؼ�����: Homo ��Ӧ�Ծ���Corners�������ĸ��ǵ�����
*********************************************************/
void ImageStitch::CalcCorners(void)
{
	double v2[] = { 0, 0, 1 };//���Ͻ�
	double v1[3];//�任�������ֵ
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //������
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //������

	V1 = Homo * V2;
	//���Ͻ�(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	Corners.left_top.x = v1[0] / v1[2];
	Corners.left_top.y = v1[1] / v1[2];

	//���½�(0,src.rows,1)
	v2[0] = 0;
	v2[1] = ImageSrc1.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = Homo * V2;
	Corners.left_bottom.x = v1[0] / v1[2];
	Corners.left_bottom.y = v1[1] / v1[2];

	//���Ͻ�(src.cols,0,1)
	v2[0] = ImageSrc1.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = Homo * V2;
	Corners.right_top.x = v1[0] / v1[2];
	Corners.right_top.y = v1[1] / v1[2];

	//���½�(src.cols,src.rows,1)
	v2[0] = ImageSrc1.cols;
	v2[1] = ImageSrc1.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //������
	V1 = Mat(3, 1, CV_64FC1, v1);  //������
	V1 = Homo * V2;
	Corners.right_bottom.x = v1[0] / v1[2];
	Corners.right_bottom.y = v1[1] / v1[2];
}
/********************************************************
���������
���������
����˵����ͼ���ںϣ������ص�����ľ��뷴�ȷ���Ȩ�أ���
		  void ImageStitch::ImageRegistration(void)����
�ؼ�������ImageSrc2��ͼ��ImageTransform1ֱ��ƥ����ͼ��
          dst �ںϺ��ͼ��
*********************************************************/
void ImageStitch::OptimizeSeam(void)
{
	int start = MIN(Corners.left_top.x, Corners.left_bottom.x);//��ʼλ�ã����ص��������߽�  

	double processWidth = ImageSrc2.cols - start;//�ص�����Ŀ��  
	int rows = dst.rows;
	int cols = ImageSrc2.cols; //ע�⣬�� ����*ͨ����(width*chanels)
	double alpha = 1;//img1�����ص�Ȩ��  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = ImageSrc2.ptr<uchar>(i);  //��ȡ��i�е��׵�ַ
		uchar* t = imageTransform1.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//�������ͼ��trans�������صĺڵ㣬����ȫ����img2�е�����
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1�����ص�Ȩ�أ��뵱ǰ�������ص�������߽�ľ���ɷ��ȣ�jԽ��Ȩ��ԽС 
				alpha = (processWidth - (j - start)) / processWidth;
			}
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
		}
	}
}
/********************************************************
���������
���������
����˵����ͼ����׼���ں�
*********************************************************/
void ImageStitch::ImageRegistration(void)
{
	if (GoodMatchePoints.size() >= 8)
	{
		//��ȡͼ��1��ͼ��2��ͶӰӳ����� �ߴ�Ϊ3*3  
		Homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	}
	else
	{
		Homo = Mat(3, 3, CV_64FC1, K);
	}

	////Ҳ����ʹ��getPerspectiveTransform�������͸�ӱ任���󣬲���Ҫ��ֻ����4���㣬Ч���Բ�  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "�任����Ϊ��\n" << Homo << endl << endl; //���ӳ�����      
	CalcCorners();
	warpPerspective(ImageSrc1, imageTransform1, Homo, Size(MAX(Corners.right_top.x, Corners.right_bottom.x), ImageSrc2.rows));
	//imwrite("trans1.jpg", imageTransform1);

	//����ƴ�Ӻ��ͼ,����ǰ����ͼ�Ĵ�С
	int dst_width = imageTransform1.cols;  //ȡ���ҵ�ĳ���Ϊƴ��ͼ�ĳ���
	int dst_height = ImageSrc1.rows;
    dst.create(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	//�����ο������Ȱѿ�����ͼ�񣬿�����ͼ��
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	ImageSrc2.copyTo(dst(Rect(0, 0, ImageSrc2.cols, ImageSrc2.rows)));
	//("direct_dst", dst);
	//ͼ���Ż��ں�
	OptimizeSeam();
}
