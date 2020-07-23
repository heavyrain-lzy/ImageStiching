#include "ImageStitch.h"


ImageStitch::ImageStitch()
{
	

}
/********************************************************
输入参数：RightPath右图像路径，LeftPath左图像路径
输出参数：
函数说明：构造函数传入图像路径
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
输入参数：
输出参数：
函数说明：是否正确读入图像
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
输入参数：
输出参数：特征点对的个数
函数说明：sift算法提取重叠区域的特征点
*********************************************************/
int ImageStitch::FindFeaturesPoints(void)
{
	//转为灰度图
	cvtColor(ImageSrc1, ImageGray1, CV_RGB2GRAY);
	cvtColor(ImageSrc2, ImageGray2, CV_RGB2GRAY);
	//提取两幅图全部的特征点
	Ptr<Feature2D> Detector = xfeatures2d::SIFT::create(2000);
	Mat imageDesc1, imageDesc2;
	
	Detector->detectAndCompute(ImageSrc1, imageDesc1, keyPoint1, imageDesc1);
	Detector->detectAndCompute(ImageSrc2, imageDesc2, keyPoint2, imageDesc2);

	//knn最邻近匹配点
	FlannBasedMatcher matcher;
	//先建立被匹配点的训练--没有分类
	vector<Mat> train_desc(1, imageDesc1);
	matcher.add(train_desc);
	matcher.train();
	matcher.knnMatch(imageDesc2, matchePoints, 2);//找出对应trainDescritors的两个匹配点，按距离排序
	cout << "total match points: " << matchePoints.size() << endl;

	// Lowe's algorithm,获取优秀匹配点
	for (int i = 0; i < matchePoints.size(); i++)
	{ 
		//ratio阈值设置为0.4获取高精度匹配点
		if (matchePoints[i][0].distance < 0.4 * matchePoints[i][1].distance)//距离最近的点的距离小于第二个的0.4
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
输入参数：
输出参数：
函数说明：显示匹配后点
*********************************************************/
void ImageStitch::FirstMatch(void)
{
	
	drawMatches(ImageSrc2, keyPoint2, ImageSrc1, keyPoint1, GoodMatchePoints, first_match);
	
}
/********************************************************
输入参数：
输出参数：
函数说明：计算右图像的四个角的通过单应性矩阵Homo转换到
		  左图像的坐标。被void ImageStitch::ImageRegistration(void)调用
关键变量: Homo 单应性矩阵；Corners计算后的四个角的坐标
*********************************************************/
void ImageStitch::CalcCorners(void)
{
	double v2[] = { 0, 0, 1 };//左上角
	double v1[3];//变换后的坐标值
	Mat V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	Mat V1 = Mat(3, 1, CV_64FC1, v1);  //列向量

	V1 = Homo * V2;
	//左上角(0,0,1)
	cout << "V2: " << V2 << endl;
	cout << "V1: " << V1 << endl;
	Corners.left_top.x = v1[0] / v1[2];
	Corners.left_top.y = v1[1] / v1[2];

	//左下角(0,src.rows,1)
	v2[0] = 0;
	v2[1] = ImageSrc1.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = Homo * V2;
	Corners.left_bottom.x = v1[0] / v1[2];
	Corners.left_bottom.y = v1[1] / v1[2];

	//右上角(src.cols,0,1)
	v2[0] = ImageSrc1.cols;
	v2[1] = 0;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = Homo * V2;
	Corners.right_top.x = v1[0] / v1[2];
	Corners.right_top.y = v1[1] / v1[2];

	//右下角(src.cols,src.rows,1)
	v2[0] = ImageSrc1.cols;
	v2[1] = ImageSrc1.rows;
	v2[2] = 1;
	V2 = Mat(3, 1, CV_64FC1, v2);  //列向量
	V1 = Mat(3, 1, CV_64FC1, v1);  //列向量
	V1 = Homo * V2;
	Corners.right_bottom.x = v1[0] / v1[2];
	Corners.right_bottom.y = v1[1] / v1[2];
}
/********************************************************
输入参数：
输出参数：
函数说明：图像融合，按到重叠区域的距离反比分配权重，被
		  void ImageStitch::ImageRegistration(void)调用
关键变量：ImageSrc2左图像，ImageTransform1直接匹配后的图像
          dst 融合后的图像
*********************************************************/
void ImageStitch::OptimizeSeam(void)
{
	int start = MIN(Corners.left_top.x, Corners.left_bottom.x);//开始位置，即重叠区域的左边界  

	double processWidth = ImageSrc2.cols - start;//重叠区域的宽度  
	int rows = dst.rows;
	int cols = ImageSrc2.cols; //注意，是 列数*通道数(width*chanels)
	double alpha = 1;//img1中像素的权重  
	for (int i = 0; i < rows; i++)
	{
		uchar* p = ImageSrc2.ptr<uchar>(i);  //获取第i行的首地址
		uchar* t = imageTransform1.ptr<uchar>(i);
		uchar* d = dst.ptr<uchar>(i);
		for (int j = start; j < cols; j++)
		{
			//如果遇到图像trans中无像素的黑点，则完全拷贝img2中的数据
			if (t[j * 3] == 0 && t[j * 3 + 1] == 0 && t[j * 3 + 2] == 0)
			{
				alpha = 1;
			}
			else
			{
				//img1中像素的权重，与当前处理点距重叠区域左边界的距离成反比，j越大权重越小 
				alpha = (processWidth - (j - start)) / processWidth;
			}
			d[j * 3] = p[j * 3] * alpha + t[j * 3] * (1 - alpha);
			d[j * 3 + 1] = p[j * 3 + 1] * alpha + t[j * 3 + 1] * (1 - alpha);
			d[j * 3 + 2] = p[j * 3 + 2] * alpha + t[j * 3 + 2] * (1 - alpha);
		}
	}
}
/********************************************************
输入参数：
输出参数：
函数说明：图像配准及融合
*********************************************************/
void ImageStitch::ImageRegistration(void)
{
	if (GoodMatchePoints.size() >= 8)
	{
		//获取图像1到图像2的投影映射矩阵 尺寸为3*3  
		Homo = findHomography(imagePoints1, imagePoints2, CV_RANSAC);
	}
	else
	{
		Homo = Mat(3, 3, CV_64FC1, K);
	}

	////也可以使用getPerspectiveTransform方法获得透视变换矩阵，不过要求只能有4个点，效果稍差  
	//Mat   homo=getPerspectiveTransform(imagePoints1,imagePoints2);  
	cout << "变换矩阵为：\n" << Homo << endl << endl; //输出映射矩阵      
	CalcCorners();
	warpPerspective(ImageSrc1, imageTransform1, Homo, Size(MAX(Corners.right_top.x, Corners.right_bottom.x), ImageSrc2.rows));
	//imwrite("trans1.jpg", imageTransform1);

	//创建拼接后的图,需提前计算图的大小
	int dst_width = imageTransform1.cols;  //取最右点的长度为拼接图的长度
	int dst_height = ImageSrc1.rows;
    dst.create(dst_height, dst_width, CV_8UC3);
	dst.setTo(0);
	//分两次拷贝，先把拷贝右图像，拷贝做图像
	imageTransform1.copyTo(dst(Rect(0, 0, imageTransform1.cols, imageTransform1.rows)));
	ImageSrc2.copyTo(dst(Rect(0, 0, ImageSrc2.cols, ImageSrc2.rows)));
	//("direct_dst", dst);
	//图像优化融合
	OptimizeSeam();
}
