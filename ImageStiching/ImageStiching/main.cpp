
#include "ImageStitch.h"
int main()
{
	//输入图像路径
	
	vector<string> LpathStr, RpathStr;
	//生成exe用代码
	LpathStr.push_back("..\\..\\question2\\L\\L000.jpg");
	LpathStr.push_back("..\\..\\question2\\L\\L001.bmp");
	LpathStr.push_back("..\\..\\question2\\L\\L002.bmp");
	LpathStr.push_back("..\\..\\question2\\L\\L004.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R000.jpg");
	RpathStr.push_back("..\\..\\question2\\R\\R001.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R002.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R004.bmp");

	////调试用代码
	//LpathStr.push_back("..\\question2\\L\\L000.jpg");
	//LpathStr.push_back("..\\question2\\L\\L001.bmp");
	//LpathStr.push_back("..\\question2\\L\\L002.bmp");
	//LpathStr.push_back("..\\question2\\L\\L004.bmp");
	//RpathStr.push_back("..\\question2\\R\\R000.jpg");
	//RpathStr.push_back("..\\question2\\R\\R001.bmp");
	//RpathStr.push_back("..\\question2\\R\\R002.bmp");
	//RpathStr.push_back("..\\question2\\R\\R004.bmp");
	//定义类
	for (int i = 0; i < RpathStr.size(); i++)
	{
		ImageStitch imageStitch(RpathStr[i], LpathStr[i]);
		bool imageFlag;
		//判断是否正确读入图像
		imageFlag = imageStitch.ReadImage();
		if (imageFlag == false)
		{
			return 1;
		}
		//特征点检测
		imageStitch.FindFeaturesPoints();
		//显示特征点
		imageStitch.FirstMatch();
		char buf[10];
		sprintf_s(buf, "%d", i);
		string firstStr="Points", dstStr="dst";
		firstStr += buf;
		dstStr += buf;
		imshow(firstStr, imageStitch.first_match);
		//图像配准及融合
		imageStitch.ImageRegistration();
		//显示最后的图像
		imshow(dstStr, imageStitch.dst);
	}
	waitKey(0);
	destroyAllWindows();
	return  0;

}