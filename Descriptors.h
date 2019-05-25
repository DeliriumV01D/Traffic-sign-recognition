#ifndef DESCRIPTORS_H
#define DESCRIPTORS_H

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/nonfree/features2d.hpp>
#include <vector>

//static const cv::Size DESCRIPTED_WINDOW_SIZE = cv::Size(128, 32);//Size(256, 56);

typedef std::vector <float> TDescriptorsVector;

///Параметры базового класса дескриптора
struct TBaseDescriptorParams {
	cv::Size descripted_window_size;
};

///Абстрактный базовый для всех дескрипторов класс
class TBaseDescriptor {
protected:
  bool GlobalDescriptior;
	TBaseDescriptorParams BaseParams;
public:
  TBaseDescriptor(const TBaseDescriptorParams &params);
  virtual ~TBaseDescriptor();
	///Задать параметры
	virtual void SetParams(const TBaseDescriptorParams &params);
  ///Размер (длина) дескриптора 
  virtual size_t GetDescriptorSize() = 0;
  ///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
  virtual cv::Size GetDescriptedWindowSize() = 0;
	///Функция вычисления глобального дескриптора
  virtual void Compute(const cv::Mat * image, cv::Mat * descriptors) = 0;
	///Функция вычисления локального дескриптора
  virtual void Compute(const cv::Rect * rect, cv::Mat * descriptors) = 0;
	///Предподготовка вычислений 
  virtual void PrepareMultiscale(cv::Mat * full_image) = 0;
	///Глобальный дескриптор или можно задать область применения
  bool IsGlobalDescriptor(){return GlobalDescriptior;};

  ///Вычисление дескриптора от картинки целиком
  void CalculateImageDescriptor(
	  const cv::Mat * image,
	  cv::Mat * descriptors,
    cv::Mat	* resized_img_buf = 0
  );
  ///Вычисление дескриптора от области предподготовленной картинки
  void CalculateImageDescriptor(
	  const cv::Rect * rect,
	  cv::Mat * descriptors
  );
  ///Получить матрицу векторов признаков рисунков каталога
	cv::Mat GetTrainingVectorFromImageCollection(const char * path);
};

///Собственно изображение как дескриптор
class TImageDescriptor : public  TBaseDescriptor{
	TBaseDescriptorParams Params;
public:
	TImageDescriptor(const TBaseDescriptorParams &params);
	~TImageDescriptor();
	///Задать параметры
	void SetParams(const TBaseDescriptorParams &params);
	///Размер (длина) дескриптора 
	size_t GetDescriptorSize();
	///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
	cv::Size GetDescriptedWindowSize();
	///Функция вычисления глобального дескриптора
	void Compute(const cv::Mat * image, cv::Mat * descriptors);
	///Не используется
	void Compute(const cv::Rect * rect, cv::Mat * descriptors){};
	///Предподготовка вычислений
	void PrepareMultiscale(cv::Mat * full_image){};
};

///Параметры дескриптора HOG
struct THOGDescriptorParams {
	TBaseDescriptorParams BaseDescriptorParams;
	cv::Size	BlockSize,		//Size(16, 16), Size(4, 4), Size(8, 8), Size(16, 16), Size(32, 32), Size(64, 64)	//меньше вертикального размера изображения
						BlockStride,	//Size(8, 8), Size(2, 2), Size(8, 8), Size(16, 16), Size(32, 32)	//меньше чем BlockSize. It must be a multiple of cell size.
						CellSize;			//Size(8, 8), Size(2, 2), Size(8, 8), Size(16, 16), Size(32, 32)	//меньше чем BlockSize
	bool GammaCorrection;		//false, false, true
};

/////Параметры по умолчанию дескриптора HOG
//const THOGDescriptorParams HOGDescriptorParamsDefaults = {
//	BaseDescriptorParamsDefaults,
//	cv::Size(24, 24),		//BlockSize
//	cv::Size(12, 12),			//BlockStride
//	cv::Size(2, 2),			//CellSize
//	false						//GammaCorrection
//};

///Histogram of Oriented Gradients дескриптор
class THOGDescriptor : public TBaseDescriptor {
protected:
  cv::HOGDescriptor * Descriptor;
	THOGDescriptorParams Params;
public:
  THOGDescriptor(
		const THOGDescriptorParams &params /*= HOGDescriptorParamsDefaults*/
	);
  ~THOGDescriptor();
	///Задать параметры алгоритма
	void SetParams(const THOGDescriptorParams &params);
  ///Размер (длина) дескриптора 
  size_t GetDescriptorSize();
  ///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
  cv::Size GetDescriptedWindowSize();
	///Функция вычисления глобального дескриптора
  void Compute(const cv::Mat * image, cv::Mat * descriptors);
	///Не используется
  void Compute(const cv::Rect * rect, cv::Mat * descriptors){};
	///Предподготовка вычислений
  void PrepareMultiscale(cv::Mat * full_image){};
};

///Параметры дескриптора GFTT
struct TGFTTDescriptorParams {
	TBaseDescriptorParams BaseDescriptorParams;
	int MaxCorners;
};

/////Параметры по умолчанию дескриптора GFTT
//const TGFTTDescriptorParams GFTTDescriptorParamsDefaults = {
//	BaseDescriptorParamsDefaults,
//	400//300!
//};

///Good Feature To Track дескриптор
class TGFTTDescriptor : public TBaseDescriptor {
protected:
  //cv::GFTTDetector * Detector;
	//cv::GridAdaptedFeatureDetector * Detector;
	cv::Ptr<cv::GFTTDetector> Detector;
  std::vector <cv::KeyPoint> Keypoints;
	TGFTTDescriptorParams Params;
public:
  TGFTTDescriptor(
		const TGFTTDescriptorParams &params /*= GFTTDescriptorParamsDefaults */
	);
  ~TGFTTDescriptor();
  ///Размер (длина) дескриптора 
  size_t GetDescriptorSize();
  ///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
  cv::Size GetDescriptedWindowSize();
	///Не используется
  void Compute(const cv::Mat * image, cv::Mat * descriptors){};
	///Функция вычисления локального дескриптора
  /**Возвращает просто общее количество особых точек GFTT*/
  void Compute(const cv::Rect * rect, cv::Mat * descriptors);
	///Предподготовка вычислений
  void PrepareMultiscale(cv::Mat * full_image);
};

///Параметры дескриптора детектора прямоугольных областей
struct TRectDetectorDescriptorParams {
	TBaseDescriptorParams BaseDescriptorParams;
	int N,									//19
			DilationSize,				//1
			MinArea;						//512
	float ApproxPrecision;	//0.04
};

///Дескриптор детектора прямоугольных областей
class TRectDetectorDescriptor : public TBaseDescriptor {
protected:
  cv::Mat Temp,
      Temp2,
      Temp3,
      Element;
  std::vector<std::vector<cv::Point> > Squares;
  std::vector<std::vector<cv::Point> > Contours;
  cv::Mat Approx;
	TRectDetectorDescriptorParams Params;
public:
  TRectDetectorDescriptor(const TRectDetectorDescriptorParams &params);
  ~TRectDetectorDescriptor();
  ///Размер (длина) дескриптора 
  size_t GetDescriptorSize(){return 8;};
  ///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
  cv::Size GetDescriptedWindowSize();
	///Не используется
  void Compute(const cv::Mat * image, cv::Mat * descriptors){};
	///Функция вычисления локального дескриптора
  void Compute(const cv::Rect * rect, cv::Mat * descriptors);
	///Предподготовка вычислений
  void PrepareMultiscale(cv::Mat * full_image);
};

/////Детектор на каскаде Хаара - срочно нужен рефакторинг TBaseCascadeDetector.AddStep(TDescriptor, TClassifier) -> 
/////TBaseCascadedetector.AddStep(TDetector(TDescriptor, TClassifier)) можно сразу добавить детектор Хаара как ступень
//class THaarDetectorDescriptor : public TBaseDescriptor {
//protected:
//	cv::CascadeClassifier Cascade;
//	std::vector<cv::Rect> Plates;
//public:
//	THaarDetectorDescriptor(const TBaseDescriptorParams &base_params = BaseDescriptorParamsDefaults);
//	~THaarDetectorDescriptor();
//	///Размер (длина) дескриптора 
//	size_t GetDescriptorSize(){ return 8; };
//	///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
//	cv::Size GetDescriptedWindowSize();
//	///Не используется
//	void Compute(cv::Mat * image, cv::Mat * descriptors){};
//	///Функция вычисления локального дескриптора
//	void Compute(cv::Rect * rect, cv::Mat * descriptors);
//	///Предподготовка вычислений
//	void PrepareMultiscale(cv::Mat * full_image);
//};

#endif
