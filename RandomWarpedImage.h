#ifndef RANDOM_WARPED_IMAGE_H
#define RANDOM_WARPED_IMAGE_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <TRandomDouble.h>

///Возвращает случайную матрицу преобразования
inline cv::Mat GetRandomWarpMatrix(
	const cv::Size &src_size,
	const float &width_variation_factor,
	const float &height_variation_factor
){
	const float delta = (float)0.03;
	cv::Mat warp_matrix;//(3,3,CV_32FC1);
	cv::Point2f src_quad[4], dst_quad[4];
		
	//Задаём точки
	src_quad[0].x = 0;
	src_quad[0].y = 0;
	src_quad[1].x = (float)src_size.width - 1;
	src_quad[1].y = 0;
	src_quad[2].x = 0;
	src_quad[2].y = (float)src_size.height - 1;
	src_quad[3].x = (float)src_size.width - 1;
	src_quad[3].y = (float)src_size.height - 1;

	dst_quad[0].x = src_size.width * (float)RandomDouble() * width_variation_factor * ((float)1 - delta);
	dst_quad[0].y = src_size.height * (float)RandomDouble() * height_variation_factor * ((float)1 - delta);
	dst_quad[1].x = src_size.width * ((float)1. - (float)RandomDouble() * width_variation_factor) * ((float)1 + delta);
	dst_quad[1].y = src_size.height * (float)RandomDouble() * height_variation_factor * ((float)1 - delta);
	dst_quad[2].x = src_size.width * (float)RandomDouble() * width_variation_factor * ((float)1 - delta);
	dst_quad[2].y = src_size.height * ((float)1. - (float)RandomDouble() * height_variation_factor) * ((float)1 + delta); 
	dst_quad[3].x = src_size.width * ((float)1. - (float)RandomDouble() * width_variation_factor) * ((float)1 + delta);
	dst_quad[3].y = src_size.height * ((float)1.- (float)RandomDouble() * height_variation_factor) * ((float)1 + delta);

	//Получаем матрицу преобразования
	warp_matrix = cv::getPerspectiveTransform(src_quad, dst_quad);
	return warp_matrix;
}

inline void WarpAndNoise(
	const cv::Mat &src, 
	cv::Mat &dst, 
	const cv::Mat warp_matrix, 	
	const float &max_brightness_variation_factor,
	const bool calc_background = false 
){
	//Вычисление среднего цвета по границе изображения
	cv::Scalar c = cvScalarAll(0);
	if (calc_background)
	{
		float c1 = 0, 
					c2 = 0, 
					c3 = 0,
					n = (float)2.*src.size().width + (float)2.*src.size().height;
		for (int i = 0; i < src.size().width; i++)
		{
			if (src.type() == CV_8UC3)
			{
				c1 += src.at<cv::Vec3b>(0, i)[0];
				c2 += src.at<cv::Vec3b>(0, i)[1];
				c3 += src.at<cv::Vec3b>(0, i)[2];

				c1 += src.at<cv::Vec3b>(src.size().height - 1, i)[0];
				c2 += src.at<cv::Vec3b>(src.size().height - 1, i)[1];
				c3 += src.at<cv::Vec3b>(src.size().height - 1, i)[2];
			};

			if (src.type() == CV_8UC1)
			{
				c1 += src.at<unsigned char>(0, i);

				c1 += src.at<unsigned char>(src.size().height - 1, i);
			};
		};

		for (int i = 0; i < src.size().height; i++)
		{
			if (src.type() == CV_8UC3)
			{
				c1 += src.at<cv::Vec3b>(i, 0)[0];
				c2 += src.at<cv::Vec3b>(i, 0)[1];
				c3 += src.at<cv::Vec3b>(i, 0)[2];

				c1 += src.at<cv::Vec3b>(i, src.size().width - 1)[0];
				c2 += src.at<cv::Vec3b>(i, src.size().width - 1)[1];
				c3 += src.at<cv::Vec3b>(i, src.size().width - 1)[2];
			};

			if (src.type() == CV_8UC1)
			{
				c1 += src.at<unsigned char>(i, 0);

				c1 += src.at<unsigned char>(i, src.size().width - 1);
			};
		};

		if (src.type() == CV_8UC3) c = cv::Scalar(c1/n, c2/n, c3/n, 0);
		if (src.type() == CV_8UC1) c = cv::Scalar(c1/n);
	};

	//Преобразование перспективы
	warpPerspective(src, dst, warp_matrix, src.size(), 1, 0, c);


	const float brightness_variation_factor = (float)RandomDouble() * max_brightness_variation_factor;
	//Вариация яркости - случайный шум
	for (int i = 0; i < dst.size().width; i++)
		for (int j = 0; j < dst.size().height; j++)
		{
			if (dst.type() == CV_8UC1)
			{
				int t = (int)dst.at<unsigned char>(j, i) + (int)(brightness_variation_factor * 256 * ((float)RandomDouble() - (float)0.5));
				if (t < 0) t = 0;
				if (t > 255) t = 255;
				dst.at<unsigned char>(j, i) = t;				
			}
			if (dst.type() == CV_8UC3)
			{
				int t;
				for (int k = 0; k < 3; k++)
				{
					t = (int)dst.at<cv::Vec3b>(j, i)[k] + (int)(brightness_variation_factor * 256 * ((float)RandomDouble() - (float)0.5));
					if (t < 0) t = 0;
					if (t > 255) t = 255;
					dst.at<cv::Vec3b>(j, i)[k] = t;
				}
			}
			if (dst.type() != CV_8UC1 && dst.type() != CV_8UC3) throw TException("GetRandomWarpedImage error: incorrect mat type");
		}
}

///Возвращает случайно перспективно трансформированное изображение
inline cv::Mat GetRandomWarpedImage(
	const cv::Mat &src,
	const float &width_variation_factor,
	const float &height_variation_factor,
	const float &brightness_variation_factor,
	const bool calc_background = false
){
	//float bf = brightness_variation_factor * (RandomDouble() - 0.5);
	cv::Mat warp_matrix = GetRandomWarpMatrix(src.size(), width_variation_factor, height_variation_factor), //(3,3,CV_32FC1);	//Получаем матрицу преобразования
					result;

	WarpAndNoise(src, result, warp_matrix, brightness_variation_factor, calc_background);
	return result;
};

#endif