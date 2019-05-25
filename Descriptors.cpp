#include <Descriptors.h>

#include <TData.h>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

/************************************************/
//TBaseDescriptor
/************************************************/ 
TBaseDescriptor :: TBaseDescriptor(const TBaseDescriptorParams &params)
{
	SetParams(params);
};

TBaseDescriptor :: ~TBaseDescriptor()
{
};

///Задать параметры
void TBaseDescriptor :: SetParams(const TBaseDescriptorParams &params)
{
	BaseParams = params;
};

//Вычисление дескриптора от картинки целиком
void TBaseDescriptor :: CalculateImageDescriptor(
	const Mat * image,
	Mat * descriptors,
  Mat	* resized_img_buf/* = 0*/
){
  bool  release_gray_img_buf = false, 
        release_resized_img_buf = false;

  if (!resized_img_buf) 
  {
    release_resized_img_buf = true;
    resized_img_buf = new Mat(GetDescriptedWindowSize(), CV_8UC1);
  };

	//Масштабируем картинку
  cv::resize(*image, *resized_img_buf, GetDescriptedWindowSize(), 0.0, 0.0, CV_INTER_LINEAR);
	//Вычисляем вектор признаков
	this->Compute(resized_img_buf, descriptors);
	if (release_resized_img_buf) delete resized_img_buf;
};

//Вычисление дескриптора от области предподготовленной картинки
void TBaseDescriptor :: CalculateImageDescriptor(
	const cv::Rect * rect,
	Mat * descriptors
){
  //Вычисляем вектор признаков
	this->Compute(rect, descriptors);
};


//Получить матрицу векторов признаков рисунков каталога
Mat TBaseDescriptor :: GetTrainingVectorFromImageCollection(
	const char * path
){
	unsigned int i = 0;
	Mat descriptors(1, (int)GetDescriptorSize(), CV_32FC1),
      gray_img_buf;
	TDataLoader * data_loader_jpg, * data_loader_bmp;
	std::list <string> :: const_iterator	current_position;

	data_loader_jpg = new TDataLoader (path, "*.jpg", true);
	data_loader_bmp = new TDataLoader (path, "*.bmp", true);

	Mat result = Mat::zeros((int)(data_loader_jpg->size() + data_loader_bmp->size()), (int)GetDescriptorSize(), CV_32FC1);
	
	//Обработка файлов "*.jpg"
	for (	current_position = data_loader_jpg->begin(); 
				current_position != data_loader_jpg->end(); 
				++current_position
			)
	{
		//cout<<*current_position<<endl;
		//Читаем исходную картинку
		gray_img_buf = cv::imread((*current_position).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		//Вычисляем дескриптор
		CalculateImageDescriptor((Mat*)&gray_img_buf, &descriptors);
		//Записываем в результирующую матрицу
		descriptors.row(0).copyTo(result.row(i));

		i++;
	};
	delete data_loader_jpg; 
	
	//Обработка файлов "*.bmp"
	for (	current_position = data_loader_bmp->begin(); 
				current_position != data_loader_bmp->end(); 
				++current_position
			)
	{
		//cout<<*current_position<<endl;
		//Читаем исходную картинку
    gray_img_buf = cv::imread((*current_position).c_str(), CV_LOAD_IMAGE_GRAYSCALE);
		//Вычисляем дескриптор
		CalculateImageDescriptor((Mat*)&gray_img_buf, &descriptors);
		//Записываем в результирующую матрицу
		descriptors.row(0).copyTo(result.row(i));

		i++;
	};
	delete data_loader_bmp; 

	return result;
};

/************************************************/
//TImageDescriptor
/************************************************/

TImageDescriptor::TImageDescriptor(const TBaseDescriptorParams &params) 
	: TBaseDescriptor(params)
{
	Params = params;
};

TImageDescriptor::~TImageDescriptor()
{
};

///Задать параметры
void TImageDescriptor::SetParams(const TBaseDescriptorParams &params)
{
	Params = params;
};

///Размер (длина) дескриптора 
size_t TImageDescriptor::GetDescriptorSize()
{
	return GetDescriptedWindowSize().area();
};

///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
cv::Size TImageDescriptor::GetDescriptedWindowSize()
{
	return Params.descripted_window_size;
};

///Функция вычисления глобального дескриптора
void TImageDescriptor::Compute(const cv::Mat * image, cv::Mat * descriptors)
{
	Mat temp;
	resize(*image, temp, GetDescriptedWindowSize());
	for (unsigned int i = 0; i < GetDescriptorSize(); i++)
		descriptors->at<float>(i) = temp.at<unsigned char>(i);
};

/************************************************/
//THOGDescriptor
/************************************************/

THOGDescriptor :: THOGDescriptor(
		const THOGDescriptorParams &params /*= HOGDescriptorParamsDefaults*/
) : TBaseDescriptor(params.BaseDescriptorParams){
	Params = params;
  GlobalDescriptior = false;
  Descriptor = new HOGDescriptor(
		BaseParams.descripted_window_size,		//win_size
		Params.BlockSize,				//block_size
    Params.BlockStride,			//block_stride
		Params.CellSize,				//cell_size
		9,									//nbins (angles)
		1,									//derivAperture
		-1.,								//win_sigma
		HOGDescriptor::L2Hys,	//histogramNormType
    0.2,								//threshold_L2hys 
		Params.GammaCorrection,							//gamma_correction
		HOGDescriptor::DEFAULT_NLEVELS			//nlevels
	);
};

THOGDescriptor :: ~THOGDescriptor()
{
  delete Descriptor;
};

//Размер (длина) дескриптора 
size_t THOGDescriptor :: GetDescriptorSize()
{
  return Descriptor->getDescriptorSize();
};

//Размер окна/изображения, описываемого дескриптором без предварительных преобразований
Size THOGDescriptor :: GetDescriptedWindowSize()
{
  return Descriptor->winSize;
};

void THOGDescriptor :: Compute(const Mat * image, Mat * descriptors)
{
  TDescriptorsVector descriptors_vector(GetDescriptorSize());
  //Mat result = Mat::zeros(GetDescriptorSize(), 1, CV_32FC1);

  Descriptor->compute(*image, descriptors_vector);
  for (unsigned int i = 0; i < descriptors_vector.size(); i++)
    descriptors->at<float>(i) = descriptors_vector[i];
};

/************************************************/
//TGFTTDescriptor
/************************************************/
TGFTTDescriptor :: TGFTTDescriptor(
	const TGFTTDescriptorParams &params /*= GFTTDescriptorDefaults*/
) : TBaseDescriptor(params.BaseDescriptorParams){
  GlobalDescriptior = true;
  //Detector = 0;
	Params = params;
};

TGFTTDescriptor :: ~TGFTTDescriptor()
{
  //if (Detector) delete Detector;
};

//Размер (длина) дескриптора 
size_t TGFTTDescriptor :: GetDescriptorSize()
{
  return 1;
};

//Размер окна/изображения, описываемого дескриптором без предварительных преобразований
Size TGFTTDescriptor :: GetDescriptedWindowSize()
{
	return BaseParams.descripted_window_size;
};

//Возвращает просто общее количество особых точек GFTT
void TGFTTDescriptor :: Compute(const cv::Rect * rect, Mat * descriptors)
{
  float result = 0.;

  for (unsigned int i = 0; i < Keypoints.size(); i++)
    if (Keypoints[i].pt.inside(*rect)) result += 1.;
  descriptors->resize(1,1);
  descriptors->at<float>(0,0) = result;
};

void TGFTTDescriptor :: PrepareMultiscale(Mat * full_image)
{
  //if (Detector) delete Detector;
	//Detector = new GridAdaptedFeatureDetector(new FastFeatureDetector(10, true)/*new cv::GFTTDetector(50)*/, full_image->rows * full_image->cols * Params.MaxCorners / (720 * 576), 4, 4);
  //Detector = new cv::GFTTDetector(
  //  full_image->rows * full_image->cols / (720 * 576) * Params.MaxCorners
  //);  //MaxCorners = 100;
	
	Detector = cv::GFTTDetector::create	(
		full_image->rows * full_image->cols / (720 * 576) * Params.MaxCorners,
		0.01,			  //double 	qualityLevel = 0.01,
		5,				  //double 	minDistance = 1,
		3					//int 	blockSize = 3,
		//bool 	useHarrisDetector = false,
		//double 	k = 0.04
	);
  Detector->detect(*full_image, Keypoints);
};

/************************************************/
//TRectDetectorDescriptor
/************************************************/

TRectDetectorDescriptor :: TRectDetectorDescriptor(const TRectDetectorDescriptorParams &params)
	: TBaseDescriptor(params.BaseDescriptorParams)
{
	Params = params;
  Element = getStructuringElement(	0,//1,2
                                    Size(2 * Params.DilationSize + 1, 2 * Params.DilationSize + 1),
                                    Point(Params.DilationSize, Params.DilationSize));
  GlobalDescriptior = true;
};

TRectDetectorDescriptor :: ~TRectDetectorDescriptor()
{};

//Размер окна/изображения, описываемого дескриптором без предварительных преобразований
cv::Size TRectDetectorDescriptor :: GetDescriptedWindowSize()
{
  return Params.BaseDescriptorParams.descripted_window_size;
};

//!!!Можно оптимизировать с помощью пространственного индекса (kdtree, например)
void TRectDetectorDescriptor :: Compute(const cv::Rect * rect, Mat * descriptors)
{ 
  unsigned int i, j;
  bool not_inside;
  Rect temp_rect;
  float square, max_square;
  Size window_size = GetDescriptedWindowSize();

  //Формирование результата
  //Прямоугольник окна
  descriptors->at<float>(0) = float(0);
  descriptors->at<float>(1) = float(0);
	descriptors->at<float>(2) = rect->width;//float(window_size.width);
	descriptors->at<float>(3) = rect->height;//float(window_size.height);

	descriptors->at<float>(4) = 0;
  descriptors->at<float>(5) = 0;
  descriptors->at<float>(6) = 0;
  descriptors->at<float>(7) = 0;
    
  //Максимальный по площади найденный прямоугольник, находящийся целиком внутри rect
  max_square = 0;
  for (i = 0; i < Squares.size(); i++)
  {
    not_inside = false;
    for (j = 0; j < Squares[i].size(); j++)
      if (!Squares[i][j].inside(*rect)) 
      {
        not_inside = true;
        break;
      };
    if (!not_inside) 
    {
      temp_rect = cv::boundingRect(cv::Mat(Squares[i]));
      //square = fabs(cv::contourArea(cv::Mat(squares[i])));
      square = float(temp_rect.area());
      if (square > max_square)
      {
        max_square = square;
        descriptors->at<float>(4) = float(temp_rect.x);
        descriptors->at<float>(5) = float(temp_rect.y);
        descriptors->at<float>(6) = float(temp_rect.width);
        descriptors->at<float>(7) = float(temp_rect.height);
      };
    };
  };  //for
};    //of Compute

void TRectDetectorDescriptor :: PrepareMultiscale(Mat * full_image)
{
  Squares.clear();

  Size window_size = GetDescriptedWindowSize();
  //Морфологическое преобразование расширения
  cv::dilate(*full_image, Temp, Element);
  //Нормализация яркости
  equalizeHist(Temp, Temp);

  for (int u = 0; u < Params.N - 2; u++)
  {
    try {
      //inRange(temp, float(u) * 255/n, float(u + 1) * 255/n, temp2);
      threshold(Temp, Temp2, float(u + 1) * 255/Params.N, 255, CV_THRESH_BINARY);
      findContours(Temp2, Contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    } catch (.../*exception &e*/) {};

    //Проверка контура на "четырехугольность"
    for (size_t i = 0; i < Contours.size(); i++)
    {
      if (Contours[i].size() < 4) continue;

      if (abs(cv::contourArea(Contours[i])) > Params.MinArea)	//Площадь взята по модулю, так как знак зависит от направления контура
      {
        //Temp3 = cv::Mat(Contours[i]);
        //Приближение контура с точностью, пропорциональной периметру
        approxPolyDP(Contours[i], Approx, arcLength(Contours[i], true) * Params.ApproxPrecision, true);
        //Контур должен содержать 4 вершины, иметь достаточную площадь и быть выпуклым
        if  ( (Approx.rows == 4) && (cv::isContourConvex(Approx)))
					Squares.push_back(Approx);
      };    //if
    }//for
  };      //for
};        //PrepareMultiscale


///************************************************/
////THaarDetectorDescriptor
///************************************************/
//
//THaarDetectorDescriptor :: THaarDetectorDescriptor(const TBaseDescriptorParams &base_params /*= BaseDescriptorParamsDefaults*/)
//{
//	BaseParams = base_params;
//	GlobalDescriptior = true;
//
//	if (Cascade.load("haarcascade_russian_plate_number.xml") == false) 
//	{
//		cout << "error while loading haarcascade_russian_plate_number.xml";
//	}
//};
//
//THaarDetectorDescriptor :: ~THaarDetectorDescriptor()
//{
//
//};
//
////Размер окна/изображения, описываемого дескриптором без предварительных преобразований
//cv::Size THaarDetectorDescriptor :: GetDescriptedWindowSize()
//{
//	return BaseParams.descripted_window_size;
//};
//
//void THaarDetectorDescriptor :: Compute(cv::Rect * rect, Mat * descriptors)
//{
//	unsigned int i;
//	bool inside;
//	Rect temp_rect;
//	//float intersection, max_intersection;
//	float square, min_square;
//	Size window_size = GetDescriptedWindowSize();
//
//	//Формирование результата
//	//Прямоугольник окна
//	descriptors->at<float>(0) = float(0);
//	descriptors->at<float>(1) = float(0);
//	descriptors->at<float>(2) = float(window_size.width);
//	descriptors->at<float>(3) = float(window_size.height);
//
//	descriptors->at<float>(4) = 0;
//	descriptors->at<float>(5) = 0;
//	descriptors->at<float>(6) = 0;
//	descriptors->at<float>(7) = 0;
//
//	//Минимальный по площади прямоугольник, включающий в себя rect
//	min_square = 0.;
//	for (i = 0; i < Plates.size(); i++)
//	{
//		inside = (Plates[i].contains(rect->tl()) && Plates[i].contains(rect->br()));
//
//		if (inside)
//		{
//			temp_rect = Plates[i];
//			square = float(temp_rect.area());
//			if ((square < min_square) || (min_square == 0.))
//			{
//				min_square = square;
//				descriptors->at<float>(4) = float(temp_rect.x);
//				descriptors->at<float>(5) = float(temp_rect.y);
//				descriptors->at<float>(6) = float(temp_rect.width);
//				descriptors->at<float>(7) = float(temp_rect.height);
//			};
//		};
//	};  //for
//
//
//	////Максимальный по площади найденный прямоугольник, находящийся целиком внутри rect
//	//max_square = 0;
//	//for (i = 0; i < Plates.size(); i++)
//	//{
//	//	inside = rect->contains(Plates[i].tl()) && rect->contains(Plates[i].br());
//
//	//	if (inside)
//	//	{
//	//		temp_rect = Plates[i];
//	//		square = float(temp_rect.area());
//	//		if (square > max_square)
//	//		{
//	//			max_square = square;
//	//			descriptors->at<float>(4) = float(temp_rect.x);
//	//			descriptors->at<float>(5) = float(temp_rect.y);
//	//			descriptors->at<float>(6) = float(temp_rect.width);
//	//			descriptors->at<float>(7) = float(temp_rect.height);
//	//		};
//	//	};
//	//};  //for
//};    //of Compute
//
//Rect GetRectIntersection(const Rect &r1, const Rect &r2)
//{
//	Rect resRect = Rect(0, 0, 0, 0);
//	Point a1 = Point(r1.x, r1.y);
//	Point a2 = Point(r1.x + r1.width, r1.y + r1.height);
//	Point b1 = Point(r2.x, r2.y);
//	Point b2 = Point(r2.x + r2.width, r2.y + r2.height);
//	//if (a1.x < b2.x && a2.x > b1.x && a1.y < b2.y && a2.y > b1.y)
//	{
//		resRect.x = max(r1.x, r2.x);
//		resRect.width = min(r1.x + r1.width, r2.width + r2.x) - resRect.x;
//		resRect.y = max(r1.y, r2.y);
//		resRect.height = min(r1.y + r1.height, r2.y + r2.height) - resRect.y;
//	}
//	return resRect;
//}
//
//void THaarDetectorDescriptor :: PrepareMultiscale(Mat * full_image)
//{
//	bool has_included;
//	int i, j;
//	Plates.clear();
//	std::vector<cv::Rect> rects;
//	//Rect intersection(0, 0, full_image->size().width, full_image->size().height);
//
//	//Детектируем 
//	Cascade.detectMultiScale(
//		*full_image,
//		rects,
//		1.1,          // scale factor
//		3,            // minimum neighbors
//		0,            // flags
//		BaseParams.descripted_window_size // minimum size
//	);
//	
//	//Оставляем только тех кандидатов, которые не содержат вложенных
//	for (i = 0; i < rects.size(); i++)
//	{
//		has_included = false;
//		for (j = 0; j < rects.size(); j++)
//			if (rects[i].contains(rects[j].br()) && rects[i].contains(rects[j].tl()))
//			{
//				has_included = true;
//				break;
//			};
//		if (!has_included)
//		{
//			Plates.push_back(rects[i]);
//			//intersection = GetRectIntersection(intersection, rects[i]);
//		}
//	};
//
//	////Если найдено ненулевое пересечение всех прямоугольников - берем его как единственный вариант
//	//if (intersection.area() > 0)
//	//{
//	//	Plates.clear();
//	//	Plates.push_back(intersection);
//	//};	
//};        //PrepareMultiscale