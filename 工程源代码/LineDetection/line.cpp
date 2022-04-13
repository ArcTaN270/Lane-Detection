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

void IamgeGraying(Mat src, Mat img)//利用加权平均法获得灰度图
{
	float R;
	float G;
	float B;

	for (int y = 0; y < src.rows; y++)
	{
		uchar* data = img.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			R = src.at<Vec3b>(y, x)[2];
			B = src.at<Vec3b>(y, x)[0];
			G = src.at<Vec3b>(y, x)[1];
			data[x] = (int)(R * 0.3 + G * 0.59 + B * 0.11);
		}
	}
}

void ThreShold(Mat src, Mat img) //灰度图二值化
{
	float start_i = 0.6 * float(src.rows); //获得0.6倍高度一下的处理图片

	int average = 0;
	for (int y = start_i; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			if (src.at<uchar>(y, x) > average) average = src.at<uchar>(y, x);
		}
	}
	average = average - 50;//通过调整阈值获得更好的检测效果

	for (int y = 0; y < src.rows; y++)
	{
		for (int x = 0; x < src.cols; x++)
		{
			uchar* data = img.ptr<uchar>(y);
			if (src.at<uchar>(y, x) > average && y >= start_i && y < src.rows)//在检测范围并且通过阈值判断如何进行二值化
			{
				data[x] = 255;
			}
			else {
				data[x] = 0;
			}
		}
	}
}

Mat GaussMark(Size size, double sigma) //根据指定滤波器大小和sigma生成滤波器
{
	Mat mask;
	mask.create(size, CV_64F);
	int h = size.height;
	int w = size.width;

	int center_h = h / 2;
	int center_w = w / 2;

	double sum = 0;
	double x;
	double y;

	int i, j;

	for (i = 0; i < h; ++i)
	{
		y = pow(i - center_h, 2);
		for (j = 0; j < w; ++j)
		{
			x = pow(j - center_w, 2);
			double s = (exp(-(x + y) / (2 * sigma * sigma))) / (2 * PI * sigma * sigma); //二维高斯分布实现
			mask.at<double>(i, j) = s;
			sum += s;
		}
	}
	mask = mask / sum; //归一化处理

	return mask;
}

Mat GaussianFilter(Mat img, Mat mark) //高斯平滑滤波处理
{
	Size size = img.size(); //图片大小
	Mat gaussian_img;
	gaussian_img = Mat::zeros(img.size(), img.type());

	int extend_h = mark.rows / 2;
	int extend_w = mark.cols / 2;

	Mat extend_img;
	copyMakeBorder(img, extend_img, extend_h, extend_h, extend_w, extend_w, BORDER_REPLICATE); //对图片边缘进行拓展以生成相同大小的图片

	int i, j;

	for (i = extend_h; i < img.rows + extend_h; i++) //图片的每一排像素进行遍历
	{
		for (j = extend_w; j < img.cols + extend_w; j++) //图像的每一列像素进行遍历
		{
			double sum[3] = { 0.0 };

			for (int r = -extend_h; r <= extend_h; r++) //滤波处理
			{
				for (int c = -extend_w; c <= extend_w; c++)
				{
					Vec3b rgb = extend_img.at<Vec3b>(i + r, j + c); //对读取的三通道图片进行处理
					sum[0] += rgb[0] * mark.at<double>(r + extend_h, c + extend_w);
					sum[1] += rgb[1] * mark.at<double>(r + extend_h, c + extend_w);
					sum[2] += rgb[2] * mark.at<double>(r + extend_h, c + extend_w);
					//sum = sum + gaussian_img.at<uchar>(i + r, j + c) * mark.at<double>(r + extend_h, c + extend_w);
				}
			}

			for (int k = 0; k < img.channels(); k++) //防止处理后数据超出[0，255]的范围
			{
				if (sum[k] < 0)
				{
					sum[k] = 0;
				}
				else if (sum[k] > 255)
				{
					sum[k] = 255;
				}
			}

			Vec3b rgb = { static_cast<uchar>(sum[0]),static_cast<uchar>(sum[1]),static_cast<uchar>(sum[2]) };
			gaussian_img.at<Vec3b>(i - extend_h, j - extend_w) = rgb;
		}
	}

	return gaussian_img;

}


void gaussian(Mat* _src, Mat mask, int _size)
{
	Mat temp = (*_src).clone();

	double sum = 0.0;
	// 遍历图片像素
	for (int i = 0; i < (*_src).rows; i++)
	{
		for (int j = 0; j < (*_src).cols; j++)
		{
			//不对边缘进行处理，避免越界的情况
			if (i > (_size / 2) - 1 && j > (_size / 2) - 1 && i < (*_src).rows - (_size / 2) && j < (*_src).cols - (_size / 2))
			{
				sum = 0.0;
				for (int k = 0; k < _size; k++)
				{
					for (int l = 0; l < _size; l++)
					{
						sum += (*_src).ptr<uchar>(i - k + (_size / 2))[j - l + (_size / 2)] * mask.at<double>(k, l);
					}
				}
				// 放入中间结果,计算所得的值与没有计算的值不能混用
				temp.ptr<uchar>(i)[j] = sum;
			}
		}
	}
	// 放入原图
	(*_src) = temp;
}


void Gaussian(Mat* src, Size size, double sigma)
{
	Mat array = GaussMark(size, sigma); //获取高斯算子数组

	int n = size.height;
	gaussian(src, array, n);
	return;
}

Mat BoundaryExtraction(Mat src) //进行边界提取
{
	Mat src_clone(src.rows, src.cols, CV_8UC1);

	for (int y = 1; y < src.rows - 1; y++)
	{
		uchar* data = src_clone.ptr<uchar>(y);
		for (int x = 1; x < src.cols - 1; x++)
		{
			int flag = 0;
			for (int i = -1; i < 2; i++)
			{
				for (int j = -1; j < 2; j++)
				{
					if (src.at<uchar>(y + i, x + j) == 0) //通过梯度对边界进行获取
					{
						flag = 1;
						break;
					}
				}
			}
			if (flag == 0)
			{
				data[x] = 255;
			}
			else data[x] = 0;
		}
	}
	//将获取的边界坐标进行还原
	for (int y = 0; y < src.rows; y++)
	{
		uchar* data = src_clone.ptr<uchar>(y);
		for (int x = 0; x < src.cols; x++)
		{
			data[x] = src.at<uchar>(y, x) - src_clone.at<uchar>(y, x);
		}
	}
	return src_clone;
}

vector<float> HoughTransform(Mat img, int threshold)
{
	int row, col;
	int i, k;

	int angle, p;//极坐标空间的参数极角angle，极径p

	//设置累加器对直线进行筛选
	int** socboard;
	int* buf;
	int w, h;
	int Size;
	int offset;

	w = img.cols;
	h = img.rows;

	vector <float> lines;
	//申请累加器空间并初始化
	Size = w * w + h * h;
	Size = 2 * sqrt(Size) + 100;
	offset = Size / 2;

	socboard = (int**)malloc(Size * sizeof(int*));
	if (!socboard)
	{
		printf("Memery apply error.\n");
		return lines;
	}

	for (i = 0; i < Size; i++)
	{
		socboard[i] = (int*)malloc(181 * sizeof(int));
		if (socboard[i] == NULL)
		{
			printf("Buffer apply error.\n");
			return lines;
		}
		memset(socboard[i], 0, 181 * sizeof(int));
	}

	//遍历图像并进行计数，得数搞得将被选中并绘制直线
	uchar src_data;
	p = 0;
	for (row = 0; row < img.rows; row++)
	{
		for (col = 0; col < img.cols; col++)
		{
			//遍历像素点
			src_data = img.at <uchar>(row, col);

			if (src_data == 255)
			{

				for (angle = 0; angle < 181; angle++)
				{
					p = col * cos(angle * PI / 180.0) + row * sin(angle * PI / 180.0) + offset;

					if (p < 0)
					{
						printf("at (%d,%d),angle:%d,p:%d\n", col, row, angle, p);
						printf("warrning!");
						printf("size:%d\n", Size / 2);
						continue;
					}
					socboard[p][angle]++;
				}
			}
		}
	}

	//遍历得数，选出的书超过阈值的直线
	int count = 0;
	int Max = 0;
	int kp, kt, r;
	kp = 0;
	kt = 0;
	for (i = 0; i < Size; i++)//对极径进行遍历
	{
		for (k = 0; k < 181; k++)//对角度进行遍历
		{
			//通过遍历所有极坐标方程对应得数进行选择
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
	Mat Mask = GaussMark(Size(5, 5), 0.8);

	vector<float>lines;

	Mat src = imread("11.jpg", 1);
	imshow("原始图", src);

	Mat src_clone = src;

	float detect_range = 0.6 * float(src.rows);

	Mat gray_scale(src.rows, src.cols, CV_8UC1);//大小与原图相同的八位单通道图
	Mat hdimg = src;

	IamgeGraying(src, gray_scale);
	imshow("灰度图", gray_scale);

	hdimg = gray_scale;
	Gaussian(&hdimg, Size(5,5), 1.5f);	
	imshow("高斯滤波后平滑图", hdimg);


	ThreShold(hdimg, gray_scale);// 灰度图二值化
	imshow("二值灰度图", gray_scale);

	gray_scale = BoundaryExtraction(gray_scale);
	imshow("边界图", gray_scale);

	lines = HoughTransform(gray_scale, 25);

	for (int i = 0; i < lines.size(); i = i + 2)
	{
		float k = lines[i];
		float b = lines[i + 1];
		if (k == 0) 
		{
			continue;
		}

		int y1 = int((detect_range - b) / k);
		int y2 = int((src.rows - b) / k);
		Point p1(y1, int(detect_range));
		Point p2(y2, int(src.rows));

		if (y1 >= 0 && y1 < src.cols && y2 >= 0 && y2 < src.cols) 
		{
			line(src_clone, p1, p2, Scalar(255, 0, 0), 2);
		}
	}

	imshow("结果图", src_clone);

	waitKey(0);
	return 0;
}