
#include "ImageStitch.h"
int main()
{
	//����ͼ��·��
	
	vector<string> LpathStr, RpathStr;
	//����exe�ô���
	LpathStr.push_back("..\\..\\question2\\L\\L000.jpg");
	LpathStr.push_back("..\\..\\question2\\L\\L001.bmp");
	LpathStr.push_back("..\\..\\question2\\L\\L002.bmp");
	LpathStr.push_back("..\\..\\question2\\L\\L004.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R000.jpg");
	RpathStr.push_back("..\\..\\question2\\R\\R001.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R002.bmp");
	RpathStr.push_back("..\\..\\question2\\R\\R004.bmp");

	////�����ô���
	//LpathStr.push_back("..\\question2\\L\\L000.jpg");
	//LpathStr.push_back("..\\question2\\L\\L001.bmp");
	//LpathStr.push_back("..\\question2\\L\\L002.bmp");
	//LpathStr.push_back("..\\question2\\L\\L004.bmp");
	//RpathStr.push_back("..\\question2\\R\\R000.jpg");
	//RpathStr.push_back("..\\question2\\R\\R001.bmp");
	//RpathStr.push_back("..\\question2\\R\\R002.bmp");
	//RpathStr.push_back("..\\question2\\R\\R004.bmp");
	//������
	for (int i = 0; i < RpathStr.size(); i++)
	{
		ImageStitch imageStitch(RpathStr[i], LpathStr[i]);
		bool imageFlag;
		//�ж��Ƿ���ȷ����ͼ��
		imageFlag = imageStitch.ReadImage();
		if (imageFlag == false)
		{
			return 1;
		}
		//��������
		imageStitch.FindFeaturesPoints();
		//��ʾ������
		imageStitch.FirstMatch();
		char buf[10];
		sprintf_s(buf, "%d", i);
		string firstStr="Points", dstStr="dst";
		firstStr += buf;
		dstStr += buf;
		imshow(firstStr, imageStitch.first_match);
		//ͼ����׼���ں�
		imageStitch.ImageRegistration();
		//��ʾ����ͼ��
		imshow(dstStr, imageStitch.dst);
	}
	waitKey(0);
	destroyAllWindows();
	return  0;

}