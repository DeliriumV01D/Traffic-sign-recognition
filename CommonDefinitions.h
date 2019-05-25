#ifndef COMMON_DEFINITIONS_H
#define COMMON_DEFINITIONS_H

#include <dlib/opencv/cv_image.h>
#include <dlib/data_io.h>
#include <opencv2/imgproc/imgproc.hpp>
#ifdef QT_VERSION
#include <string>
#include <QtCore/QDebug>
#include <QtGui/QImage>
#include <QtGui/QPixmap>

#include <opencv2/imgproc/types_c.h>

#include <WriteToLog.h>

///ѕреобразование cv::Mat в QImage
inline QImage cvMatToQImage(const cv::Mat &mat)
{
  switch (mat.type())
  {
    // 8-bit, 4 channel
    case CV_8UC4:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_ARGB32);
      return image;
    }

    // 8-bit, 3 channel
    case CV_8UC3:
    {
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_RGB888);
      return image.rgbSwapped();
    }

    // 8-bit, 1 channel
    case CV_8UC1:
    {
      #if QT_VERSION >= QT_VERSION_CHECK(5, 5, 0)
      QImage image(mat.data, mat.cols, mat.rows, static_cast<int>(mat.step), QImage::Format_Grayscale8);
      #else
      static QVector<QRgb>  sColorTable;
      // only create our color table the first time
      if (sColorTable.isEmpty())
      {
        sColorTable.resize(256);
        for (int i = 0; i < 256; ++i) sColorTable[i] = qRgb(i, i, i);
      }
      QImage image(inMat.data, inMat.cols, inMat.rows, static_cast<int>(inMat.step), QImage::Format_Indexed8);
      image.setColorTable(sColorTable);
      #endif
      return image;
    }

    default:
      throw TException("cvMatToQImage() error: cv::Mat image type not supported: " /*+ std::to_string(mat.type())*/);
    break;
  }
  return QImage();
}

///ѕреобразование cv::Mat в QPixmap
inline QPixmap cvMatToQPixmap(const cv::Mat &mat)
{
  return QPixmap::fromImage(cvMatToQImage(mat));
}

#endif  //#ifdef QT_VERSION

inline void rotateImage(const cv::Mat &in, cv::Mat &out, const double &angle)
{
  // get rotation matrix for rotating the image around its center
  cv::Point2f center(in.cols/2.0, in.rows/2.0);
  cv::Mat rot = cv::getRotationMatrix2D(center, angle, 1.0);
  // determine bounding rectangle
  cv::Rect bbox = cv::RotatedRect(center, in.size(), angle).boundingRect();
  // adjust transformation matrix
  rot.at<double>(0,2) += bbox.width/2.0 - center.x;
  rot.at<double>(1,2) += bbox.height/2.0 - center.y;

  cv::warpAffine(in, out, rot, bbox.size());
}

///ѕреобразование в градации серого
static void ToGrayscale(const cv::Mat &image, cv::Mat &gray)
{
	if (image.type() != CV_8UC1)
	{
		cv::cvtColor(image, gray, CV_BGR2GRAY);
		gray.convertTo(gray, CV_8UC1);
	} else {
		image.copyTo(gray);
	}
}

///ѕреобразование матрицы чисел с плавающей зап€той из формата opencv в формат dlib
static void CVMatToDlibMatrixFC1(const cv::Mat &mat, dlib::matrix<float> &dlib_matrix)
{
	cv::Mat temp(mat.cols, mat.rows, CV_32FC1);
	cv::normalize(mat, temp, 0.0, 1.0, cv::NORM_MINMAX, CV_32FC1);
	dlib::assign_image(dlib_matrix, dlib::cv_image<float>(temp));
}

///ѕреобразование матрицы целых чисел из формата opencv в формат dlib
static void CVMatToDlibMatrix8U(const cv::Mat &mat, dlib::matrix<unsigned char> &dlib_matrix)
{
	cv::Mat temp(mat.cols, mat.rows, CV_8U);
	cv::normalize(mat, temp, 0, 255, cv::NORM_MINMAX, CV_8U);
	dlib::assign_image(dlib_matrix, dlib::cv_image<unsigned char>(temp));
}

#endif