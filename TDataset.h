#ifndef TDATASET_H
#define TDATASET_H

#include <CommonDefinitions.h>
#include <TRandomInt.h>
#include <TIndexedObjects.h>
#include <TData.h>
#include <string>
#include <mutex>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>

///Вспомогательная структура для хранения адресов элементов выборки и их меток
class TLabeledSamplePath {
public:
	std::string SamplePath;
	unsigned long Label;

	TLabeledSamplePath(const std::string sample_path, const unsigned long label)
	{
		SamplePath = sample_path;
		Label = label;
	};
};

///Параметры датасета для работы с обучающей выборкой
/*typedef*/ struct TDatasetProperties {
 	std::string Dir;
	cv::Size ImgSize;
	bool	UseMultiThreading,
				OneThreadReading,
				Warp;
};

///Параметры датасета для работы с обучающей выборкой по умолчанию
static const TDatasetProperties DATASET_PROPERTIES_DEFAULTS = 
{
	"",
	cv::Size(0, 0),
	true,
	false,
	false
};

///Датасет для работы с обучающей выборкой
class TDataset {
protected:
	TDatasetProperties DatasetProperties;
	TIndexedObjects <std::string> Labels;										///Метки классов - имена подкаталогов, текст из файлов
	std::vector <TLabeledSamplePath> LabeledSamplePaths;		///Адреса элементов обучающей выборки и метки классов
	dlib::thread_pool * ThreadPool;
	std::mutex	ReadMutex,																	///На случай one_thread_reading закроет imread
							DetectMutex;
public:
	TDataset(const TDatasetProperties &dataset_properties);
	virtual ~TDataset();

	virtual unsigned long Size();
	virtual unsigned long ClassNumber();

	///Работаем в предположении, что все изображения одного объекта лежат в одном подкаталоге
	virtual std::string GetLabelFromPath(const std::string &path);
	///Получить объект из выборки по индексу
	virtual cv::Mat GetSampleCVMat(const unsigned long sample_idx);
	///Получить объект из выборки по индексу
	virtual dlib::matrix<unsigned char> GetSampleDLibMatrix(const unsigned long sample_idx);
	///Получить метку по индексу объекта
	virtual std::string GetLabel(const unsigned long sample_idx);
	///Получить метку по индексу метки
	virtual std::string GetLabelByIdx(const unsigned long label_idx);
	///Получить индекс метки из выборки по индексу
	virtual unsigned long GetLabelIdx(const unsigned long sample_idx);
	///Сформировать вход нейросетки по двум изображениям(одинакового размера!)
	virtual dlib::matrix<unsigned char> MakeInputSamplePair(dlib::matrix<unsigned char> * img1, dlib::matrix<unsigned char> * img2);

	///Получить пару изображений positive == true одного объекта, false - разных объектов и соответствующую метку
	virtual void GetInputSamplePair(bool positive, dlib::matrix<unsigned char> &sample_pair, unsigned long &label);
	///Получить пакет пар изображений и соответвующие метки
	virtual void GetInputSamplePairBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_sample_pairs, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);

	///Случайный объект из датасета и соответствующая ему метка	 cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSample(dlib::matrix<unsigned char> &sample, unsigned long &label);
	///Случайный объект из датасета и соответствующая ему метка  cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSample(cv::Mat &sample, unsigned long &label);
	///Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
	virtual void GetRandomSampleBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_samples, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);
	/////Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
	//virtual void GetRandomSampleBatch(
	//	cv::Mat &data, 
	//	cv::Mat &classes,
	//	const size_t batch_size
	//);
};

#endif