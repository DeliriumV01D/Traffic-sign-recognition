#ifndef CLASSIFIERS_H
#define CLASSIFIERS_H

#include <opencv/ml.h>
#include <WriteToLog.h>

///Интерфейсный базовый класс классификатора
class TBaseClassifier {
protected:
public:
	TBaseClassifier(){};
	virtual ~TBaseClassifier(){};

  ///Обучение 
  virtual void Train(const cv::Mat &train_data, const cv::Mat &train_classes) = 0;
  ///Сохранение параметров
  virtual void Save(const std::string &file_name) = 0;
  ///Загрузка параметров
  virtual void Load(const std::string &file_name) = 0;
  ///Классификация
  virtual float Predict(const cv::Mat &samples, cv::Mat &results) = 0;
};

///Параметры классификатора, основанного на методе опорных векторов
struct TSVMClassifierParams {
	long KernelType;		//cv::SVM::LINEAR, LINEAR, POLY, RBF, SIGMOID
	double	Degree,
					Gamma;
};

///Параметры по умолчанию классификатора, основанного на методе опорных векторов
const TSVMClassifierParams SVM_CLASSIFIER_PARAMS_DEFAULTS = 
{
	cv::ml::SVM::LINEAR,			//KernelType
	3,												//Degree
	3													//Gamma
};

///Классификатор, основанный на методе опорных векторов
class TSVMClassifier : public TBaseClassifier {
protected:
  cv::Ptr<cv::ml::SVM> Classifier;
  TSVMClassifierParams SVMClassifierParams;
public:
  TSVMClassifier(const TSVMClassifierParams &params = SVM_CLASSIFIER_PARAMS_DEFAULTS);
  virtual ~TSVMClassifier();

	///Задать параметры алгоритма
  virtual void SetParams(const TSVMClassifierParams &params);
  ///Обучение 
  virtual void Train(const cv::Mat &train_data, const cv::Mat &train_classes);
  ///Сохранение параметров
  virtual void Save(const std::string &file_name);
  ///Загрузка параметров
  virtual void Load(const std::string &file_name);
  ///Классификация
	float Predict(const cv::Mat &samples, cv::Mat &results);
};

///Классификатор, основанный на методе опорных векторов с предварительным сжатием методом главных компонент
class TPCA_SVMClassifier : public TSVMClassifier {
protected:
	int PCASetSize,
			MaxComponents;
	cv::Mat PCASet;
	cv::PCA * Pca;
public:
	TPCA_SVMClassifier(
		const TSVMClassifierParams &params = SVM_CLASSIFIER_PARAMS_DEFAULTS, 
		const long pca_set_size = 10000,
		const long max_components = 0
	);
	~TPCA_SVMClassifier();

	///Задать параметры алгоритма
  void SetParams(
		const TSVMClassifierParams &params,
		const long pca_set_size = 10000,
		const long max_components = 0
	);
  ///Обучение 
  void Train(const cv::Mat &train_data, const cv::Mat &train_classes);
  ///Сохранение параметров
  void Save(const std::string &file_name);
  ///Загрузка параметров
  void Load(const std::string &file_name);
  ///Классификация
	float Predict(const cv::Mat &samples, cv::Mat &results);
};

///Параметры классификатора, основанного на методе k-ближайших соседей
struct TKNNClassifierParams {
	bool IsClassifier;		//or Regression
	int AlgorithmType,		//        BRUTE_FORCE=1, KDTREE=2
			DefaultK;
};

///Параметры по умолчанию классификатора, основанного на методе опорных векторов
const TKNNClassifierParams KNN_CLASSIFIER_PARAMS_DEFAULTS = 
{
	0,																				//IsClassifier
	cv::ml::KNearest::BRUTE_FORCE,						//AlgorithmType
	1																					//DefaultK
};

///Классификатор, основанный на методе k-ближайших соседей
class TKNNClassifier : public TBaseClassifier {
protected:
  cv::Ptr<cv::ml::KNearest> Classifier;
  TKNNClassifierParams KNNClassifierParams;
public:
  TKNNClassifier(const TKNNClassifierParams &params = KNN_CLASSIFIER_PARAMS_DEFAULTS);
  virtual ~TKNNClassifier();

	///Задать параметры алгоритма
  virtual void SetParams(const TKNNClassifierParams &params);
  ///Обучение 
  virtual void Train(const cv::Mat &train_data, const cv::Mat &train_classes);
  ///Сохранение параметров
  virtual void Save(const std::string &file_name);
  ///Загрузка параметров
  virtual void Load(const std::string &file_name);
  ///Классификация
	float Predict(const cv::Mat &samples, cv::Mat &results);
};

#endif
