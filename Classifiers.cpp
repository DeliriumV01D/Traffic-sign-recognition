#include <Classifiers.h>
using namespace cv;

/************************************************/
//TSVMClassifier
//Классификатор, основанный на методе опорных векторов
/************************************************/ 

TSVMClassifier :: TSVMClassifier (const TSVMClassifierParams &params /*= SVMClassifierParamsDefaults*/) : TBaseClassifier ()
{
	SetParams(params);
};

TSVMClassifier :: ~TSVMClassifier ()
{
};

//Задать параметры алгоритма
void TSVMClassifier :: SetParams(const TSVMClassifierParams &params)
{
	SVMClassifierParams = params;
};

//Обучение классификаторов
void TSVMClassifier :: Train(
  const cv::Mat &train_data, 
  const cv::Mat &train_classes
){
	//Устанавливаем тип алгоритма и ядра в параметрах
	Classifier = cv::ml::SVM::create();
	Classifier->setKernel(SVMClassifierParams.KernelType); //LINEAR, POLY, RBF, SIGMOID
	Classifier->setType(cv::ml::SVM::C_SVC);
	if (SVMClassifierParams.KernelType == cv::ml::SVM::POLY)
	{
		Classifier->setDegree(SVMClassifierParams.Degree);
		Classifier->setGamma(SVMClassifierParams.Gamma);
	};
	Classifier->train(train_data, cv::ml::ROW_SAMPLE, train_classes);
};

//Сохранение параметров
void TSVMClassifier :: Save(const std::string &file_name)
{
  Classifier->save(file_name.c_str());
};

//Загрузка параметров
void TSVMClassifier :: Load(const std::string &file_name)
{
	Classifier = cv::Algorithm::load<cv::ml::SVM>(file_name.c_str());
};

//Классификация
float TSVMClassifier :: Predict(const cv::Mat &samples, cv::Mat &results)
{
	return Classifier->predict(samples, results);
};

/************************************************/
//TPCA_SVMClassifier
//Классификатор, основанный на методе опорных векторов с предварительным сжатием методом главных компонент
/************************************************/
TPCA_SVMClassifier :: TPCA_SVMClassifier(
	const TSVMClassifierParams &params /*= SVMClassifierParamsDefaults*/, 
	const long pca_set_size /*= 1000*/,
	const long max_components /*= 0*/
){
	Pca = 0;
	SetParams(params, pca_set_size, max_components);
};

TPCA_SVMClassifier :: ~TPCA_SVMClassifier()
{
	if (Pca) delete Pca;
};

//Задать параметры алгоритма
void TPCA_SVMClassifier :: SetParams(		
	const TSVMClassifierParams &params,
	const long pca_set_size /*= 1000*/,
	const long max_components /*= 0*/
){
	PCASetSize = pca_set_size;
	MaxComponents = max_components;
	TSVMClassifier::SetParams(params);
};

//Обучение 
void TPCA_SVMClassifier :: Train(const cv::Mat &train_data, const cv::Mat &train_classes)
{
	int i, i_pca_set = 0, i_obj = 0, n_obj = 0;

  if (MaxComponents == 0) MaxComponents = train_data.cols/4;

  cv::Mat pca_train_data(train_data.rows, MaxComponents, train_data.type()),
          vec,
          coeffs;

  for (i = 0; i < train_classes.rows; i++)
    if (train_classes.at<int>(i) == 1) n_obj++;

  if (PCASetSize > n_obj) PCASetSize = n_obj;

  PCASet.create(PCASetSize, train_data.cols, train_data.type());

  //Формирование выборки для вычисления главных компонент
  for (i = 0; i < train_data.rows; i++)
    if ((train_classes.at<int>(i) == 1))
    {
      if ((i_obj % int(n_obj/PCASetSize) == 0) &&
        (i_obj != 0) &&
        (i_pca_set < PCASetSize)
      ){
        train_data.row(i).copyTo(PCASet.row(i_pca_set));
        i_pca_set++;
      };

      i_obj++;
    };

	//Вычисление собственных векторов и собственных значений матрицы данных (главных компонент)
	if (Pca) delete Pca;
	Pca = new cv::PCA(	PCASet,				//Данные
											cv::Mat(),		//Не имеем заранее вычисленного среднего, пусть сам считает
											cv::PCA::DATA_AS_ROW,						//cv::PCA::DATA_AS_ROW, //Определение типа укладки векторов в матрицу данных
											MaxComponents	//Число главных компонент, которые собираемся использовать
									 );

	PCASet.release();

	//Вычисление коэффициентов разложения по первым mc собственным векторам, упорядоченным по убыванию собственных значений
	for (i = 0; i < train_data.rows; i++)
	{
		vec = train_data.row(i);
		Pca->project(vec, coeffs);
		coeffs.copyTo(pca_train_data.row(i));
	};
		
	TSVMClassifier::Train(pca_train_data, train_classes);
};

//Сохранение параметров
void TPCA_SVMClassifier :: Save(const std::string &file_name)
{
	//Сохранение параметров МГК
  cv::FileStorage fs(file_name+".pca", cv::FileStorage::WRITE);

  Pca->write(fs);
  fs.release();

  //Сохранение опорных векторов
  TSVMClassifier::Save(file_name);
};

//Загрузка параметров
void TPCA_SVMClassifier :: Load(const std::string &file_name)
{
	//Загрузка параметров МГК
  cv::FileStorage fs(file_name+".pca", cv::FileStorage::READ);

  if (Pca) delete Pca;
  Pca = new cv::PCA();

  Pca->read(fs.root());

  //Загрузка опорных векторов
  TSVMClassifier::Load(file_name);
};

//Классификация
float TPCA_SVMClassifier :: Predict(const cv::Mat &samples, cv::Mat &results)
{
	cv::Mat coeffs;
	Pca->project(samples, coeffs);
	return TSVMClassifier::Predict(coeffs, results);
};

/************************************************/
//TKNNClassifier
//Классификатор, основанный на методе k-ближайших соседей
/************************************************/ 

TKNNClassifier :: TKNNClassifier (const TKNNClassifierParams &params /*= KNNClassifierDefaults*/) : TBaseClassifier ()
{
	SetParams(params);
};

TKNNClassifier :: ~TKNNClassifier ()
{
};

//Задать параметры алгоритма
void TKNNClassifier :: SetParams(const TKNNClassifierParams &params)
{
	KNNClassifierParams = params;
};

//Обучение классификаторов
void TKNNClassifier :: Train(
  const cv::Mat &train_data, 
  const cv::Mat &train_classes
){
	Classifier = cv::ml::KNearest::create();
	Classifier->setIsClassifier(KNNClassifierParams.IsClassifier);
	Classifier->setAlgorithmType(KNNClassifierParams.AlgorithmType);
	Classifier->setDefaultK(KNNClassifierParams.DefaultK);

	cv::Ptr<cv::ml::TrainData> train_data_ptr;
	train_data_ptr = cv::ml::TrainData::create(train_data, cv::ml::ROW_SAMPLE, train_classes);
	Classifier->train(train_data_ptr);
};

//Сохранение параметров
void TKNNClassifier :: Save(const std::string &file_name)
{
  Classifier->save(file_name.c_str());
};

//Загрузка параметров
void TKNNClassifier :: Load(const std::string &file_name)
{
	Classifier = cv::Algorithm::load<cv::ml::KNearest>(file_name.c_str());
};

//Классификация
float TKNNClassifier :: Predict(const cv::Mat &samples, cv::Mat &results)
{
	Classifier->findNearest(samples, Classifier->getDefaultK(), results /*, distance*/);
	return 0;
};