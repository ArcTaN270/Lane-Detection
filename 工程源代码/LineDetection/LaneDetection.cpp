#include<iostream>  
#include<math.h>
#include<string.h>
#include<stdlib.h>
#include <memory.h>
#include <vector>
#include<opencv2/opencv.hpp>
#define PI 3.1415926
using namespace std;
using namespace cv;

// Using average method to transform the image into grayscale image
void cvtCOLOR(Mat src, Mat img)
{
	float R;
	float G;
	float B;

	for (int y = 0; y < src.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			R = src.at<Vec3b>(y, x)[2];
			data[x] = (int)(R * 0.299 + G * 0.587 + B * 0.114);
		}
	}
}

Mat HistogramEqualization(Mat image) {
	double gray[256] = { 0 };

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			int k = image.at<uchar>(i, j);
			gray[k]++;
		}
	}

	for (int x = 0; x < 256; x++) {

		gray[x] = (double)gray[x] / (image.rows * image.cols);
	}

	for (int m = 0; m < 256; m++) {
		gray[m + 1] = gray[m + 1] + gray[m];
	}

	for (int a = 0; a < image.rows; a++) {
		for (int b = 0; b < image.cols; b++) {
			int k = image.at<uchar>(a, b);
			image.at<uchar>(a, b) = (uchar)(255 * gray[k] + 0.5);
		}
	}
	return image;
}  //直方图均衡化

void ThreShold(Mat src , Mat dst) { //灰度图二值化
	float start_i = 0.6 * float(src.rows);
	float end_i = 1 * float(src.rows);
	int average = 0;
	for (int y = start_i; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<uchar>(y, x) > average) average = src.at<uchar>(y, x);
		}
	}
	average = average - 60;

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			uchar* data = dst.ptr<uchar>(y);
			if (src.at<uchar>(y, x) > average && y >= start_i && y < end_i)
			{
				data[x] = 255;
			}
			else {
				data[x] = 0;
			}
		}
	}
}

/* 高斯滤波 (待处理单通道图片, 高斯分布数组， 高斯数组大小(核大小) ) */
void gaussian(cv::Mat* _src, double** _array, int _size)
{
	cv::Mat temp = (*_src).clone();
	// [1] 扫描
	for (int i = 0; i < (*_src).rows; i++) {
		for (int j = 0; j < (*_src).cols; j++) {
			// [2] 忽略边缘
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 &&
				i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2)) {
				// [3] 找到图像输入点f(i,j),以输入点为中心与核中心对齐
				//     核心为中心参考点 卷积算子=>高斯矩阵180度转向计算
				//     x y 代表卷积核的权值坐标   i j 代表图像输入点坐标
				//     卷积算子     (f*g)(i,j) = f(i-k,j-l)g(k,l)          f代表图像输入 g代表核
				//     带入核参考点 (f*g)(i,j) = f(i-(k-ai), j-(l-aj))g(k,l)   ai,aj 核参考点
				//     加权求和  注意：核的坐标以左上0,0起点
				double sum = 0.0;
				for (int k = 0; k < _size; k++) {
					for (int l = 0; l < _size; l++) {
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * _array[k][l];
					}
				}
				// 放入中间结果,计算所得的值与没有计算的值不能混用
				temp.ptr<uchar>(i)[j] = sum;
			}
		}
	}

	// 放入原图
	(*_src) = temp.clone();
}

double** getGaussianArray(int arr_size, double sigma)
{
	int i, j;
	// [1] 初始化权值数组
	double** array = new double* [arr_size];
	for (i = 0; i < arr_size; i++) {
		array[i] = new double[arr_size];
	}
	// [2] 高斯分布计算
	int center_i, center_j;
	center_i = center_j = arr_size / 2;
	double pi = 3.141592653589793;
	double sum = 0.0f;
	// [2-1] 高斯函数
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] =
				//后面进行归一化，这部分可以不用
				//0.5f *pi*(sigma*sigma) * 
				exp(-(1.0f) * (((i - center_i) * (i - center_i) + (j - center_j) * (j - center_j)) /
					(2.0f * sigma * sigma)));
			sum += array[i][j];
		}
	}
	// [2-2] 归一化求权值
	for (i = 0; i < arr_size; i++) {
		for (j = 0; j < arr_size; j++) {
			array[i][j] /= sum;
			printf(" [%.15f] ", array[i][j]);
		}
		printf("\n");
	}
	return array;
}

void myGaussianFilter(cv::Mat* src, cv::Mat* dst, int n, double sigma)
{
	// [1] 初始化
	*dst = (*src).clone();
	// [2] 滤波
	// [2-1] 确定高斯正态矩阵
	double** array = getGaussianArray(n, sigma);
	// [2-2] 高斯滤波处理
	gaussian(src, array, n);
	return;
}

Mat BoundaryExtraction(Mat src) {
	Mat dst(src.rows, src.cols, CV_8UC1);
	for (int y = 1; y < src.rows-1; y++)
	{
		uchar* data = dst.ptr<uchar>(y);
		for (int x = 1; x < src.cols-1; x++)
		{
			int flag = 0;
			for (int i = -1; i < 2; i++) {
				for (int j = -1; j < 2; j++) {
					if (src.at<uchar>(y + i, x + j) == 0) {
						flag = 1;
						break;
					}
				}
			}
			if (flag == 0) data[x] = 255;
			else data[x] = 0;
		}
	}
	for (int y = 0; y < src.rows; y++)
	{
		uchar* data = dst.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			data[x] = src.at<uchar>(y, x) - dst.at<uchar>(y, x);
		}
	}
	return dst;
}

vector<float> hough_line_v(Mat img, int threshold)
{
	int row, col;
	int i, k;
	//参数空间的参数极角angle(角度)，极径p;
	int angle, p;

	//累加器
	int** socboard;
	int* buf;
	int w, h;
	w = img.cols;
	h = img.rows;
	int Size;
	int offset;

	vector<float> lines;
	//申请累加器空间并初始化
	Size = w * w + h * h;
	Size = 2 * sqrt(Size) + 100;
	offset = Size / 2;
	socboard = (int**)malloc(Size * sizeof(int*));
	if (!socboard)
	{
		printf("mem err\n");
		return lines;
	}

	for (i = 0; i < Size; i++)
	{
		socboard[i] = (int*)malloc(181 * sizeof(int));
		if (socboard[i] == NULL)
		{
			printf("buf err\n");
			return lines;
		}
		memset(socboard[i], 0, 181 * sizeof(int));
	}

	//遍历图像并投票
	uchar src_data;
	p = 0;
	for (row = 0; row < img.rows; row++)
	{
		for (col = 0; col < img.cols; col++)
		{
			//获取像素点
			src_data = img.at <uchar>(row, col);

			if (src_data == 255)
			{

				for (angle = 0; angle < 181; angle++)
				{
					p = col * cos(angle * PI / 180.0) + row * sin(angle * PI / 180.0) + offset;

					//错误处理
					if (p < 0)
					{
						printf("at (%d,%d),angle:%d,p:%d\n", col, row, angle, p);
						printf("warrning!");
						printf("size:%d\n", Size / 2);
						continue;
					}
					//投票计分
					socboard[p][angle]++;

				}
			}
		}
	}

	//遍历计分板，选出符合阈值条件的直线
	int count = 0;
	int Max = 0;
	int kp, kt, r;
	kp = 0;
	kt = 0;
	for (i = 0; i < Size; i++)//p
	{
		for (k = 0; k < 181; k++)//angle
		{
			if (socboard[i][k] > Max)
			{
				Max = socboard[i][k];
				kp = i - offset;
				kt = k;
			}

			if (socboard[i][k] >= threshold)
			{
				r = i - offset;
				lines.push_back(-1.0 * float(std::cos(k * PI / 180) / std::sin(k * PI / 180)));
				lines.push_back(float(r) / std::sin(k * PI / 180));
				count++;
			}
		}
	}
	//释放资源
	for (int e = 0; e < Size; e++)
	{
		free(socboard[e]);
	}
	free(socboard);
	return lines;
}


int main()
{
	vector<float>lines;
	const char* filename = "3.jpg";
	Mat src = imread(filename, 1);
	Mat src1(src.rows, src.cols, CV_8UC3);
	float start_i = 0.7 * float(src.rows);
	src1 = src;
	Mat dst(src.rows, src.cols, CV_8UC1);//大小与原图相同的八位单通道图
	Mat dst1(src.rows, src.cols, CV_8UC1);
	Mat yuan = src;
	imshow("原始图", src);
	cvtCOLOR(src, dst); //彩色图转换灰度图
	yuan = dst;
	dst = dst1;
//	dst = HistogramEqualization(yuan);
//	yuan = dst;
//	dst = dst1;
	myGaussianFilter(&yuan, &dst, 5, 1.5f);		//高斯滤波降噪
	//imshow("灰度图", dst);
	yuan = dst;
	dst = dst1;
	ThreShold(yuan, dst);// 灰度图二值化
	yuan = dst;
	dst = dst1;
	myGaussianFilter(&yuan, &dst, 5, 1.5f);		//高斯滤波降噪
	//imshow("二值图", dst);
	yuan = dst;
	dst = dst1;
	dst = BoundaryExtraction(yuan);
	yuan = dst;
	dst = dst1;
	myGaussianFilter(&yuan, &dst, 5, 1.5f);
	//imshow("边界图", dst);
	yuan = dst;
	lines = hough_line_v(yuan, 25);  //40 25
	
	for (int i = 0; i < lines.size(); i = i + 2) {
		float k = lines[i];
		float b = lines[i + 1];
		if (k == 0) {
			continue;
		}
	//	printf("%f %f\n", k, b);
		int y1 = int((start_i - b) / k);
		int y2 = int((src1.rows - b) / k);
		cv::Point p1(y1, int(start_i));
		cv::Point p2(y2, int(src1.rows));
		if (y1 >= 0 && y1 < src1.cols && y2 >= 0 && y2 < src1.cols) {
			line(src1, p1, p2, cv::Scalar(0, 255, 0), 2);
		}
	}

	imshow("结果图", src1);
//	imshow("灰度图", dst);
	waitKey(0);
	return 0;
}