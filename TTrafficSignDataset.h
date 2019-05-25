#ifndef TTRAFFIC_SIGN_DATASET_H
#define TTRAFFIC_SIGN_DATASET_H

#include <TDataset.h>
#include <multiscale_sampler.h>
#include <TRandomDouble.h>

///Параметры датасета для работы с обучающей выборкой
/*typedef*/ struct TTrafficSignDatasetProperties {
	TDatasetProperties	ObjDatasetProperties,
											BackDatasetProperties;
	double Ratio;
	unsigned long BackLabel;
};

///Параметры датасета для работы с обучающей выборкой по умолчанию
static const TTrafficSignDatasetProperties TRAFFIC_SIGN_DATASET_PROPERTIES_DEFAULTS = 
{
	DATASET_PROPERTIES_DEFAULTS,
	DATASET_PROPERTIES_DEFAULTS,
	0.1,
	777
};

///Генерирует пакеты обучающей выборки, комбинируя дополненные изображения объектов и фрагменты фоновых изображений
class	TTrafficSignDataset /*: public TDataset*/ {
protected:
	TTrafficSignDatasetProperties TSDatasetProperties;
	TDataset	* ObjDataset,
						* BackDataset;
	MultiscaleSampler * MultiSampler;
	unsigned long BackIdx,
								BackLocationIdx;
	cv::Mat CurrentBackImage;
	std::vector <cv::Rect> CurrentBackLocations;
public:
	TTrafficSignDataset(const TTrafficSignDatasetProperties &properties)
		//: TDataset(properties.DatasetProperties)
	{
		TSDatasetProperties = properties;
		ObjDataset = new TDataset(TSDatasetProperties.ObjDatasetProperties);
		BackDataset = new TDataset(TSDatasetProperties.BackDatasetProperties);
		MultiSampler = new MultiscaleSampler(TSDatasetProperties.BackDatasetProperties.ImgSize);
		BackIdx = 0;
	};
	virtual ~TTrafficSignDataset()
	{
		delete MultiSampler;
		delete BackDataset;
		delete ObjDataset;
	};

	///Получить очередной объект из выборки	и соответствующую ему метку
	void GetSample(cv::Mat &sample, unsigned long &label)
	{
		if (RandomDouble() <= TSDatasetProperties.Ratio)
		{
			ObjDataset->GetRandomSample(sample, label);
		}else{
			if (CurrentBackLocations.size() == 0 || BackLocationIdx >= CurrentBackLocations.size())
			{
				std::cout<<CurrentBackLocations.size()<<std::endl;
				CurrentBackImage = BackDataset->GetSampleCVMat(BackIdx);
				MultiSampler->getSamples(CurrentBackImage, std::vector <cv::Rect>(), CurrentBackLocations);

				BackLocationIdx = 0;

				BackIdx++;
				if (BackIdx == BackDataset->Size())
					BackIdx = 0;
			}

			sample = CurrentBackImage(CurrentBackLocations[BackLocationIdx]);
			label = TSDatasetProperties.BackLabel;
			BackLocationIdx++;
		}
	}

	///
	void GetSample(dlib::matrix<unsigned char> &sample, unsigned long &label)
	{
		cv::Mat mat_sample;
		GetSample(mat_sample, label);
		//Преобразуем в dlib::matrix
		CVMatToDlibMatrix8U(mat_sample, sample);
	}

	//Получить пакет изображений и соответствующие им метки  cv::Mat - dlib::matrix переписать через шаблон?
	void GetSampleBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_samples, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	) {
		cv::Mat gray_img_buf;
		dlib::matrix<unsigned char> temp, 
																sample(TSDatasetProperties.ObjDatasetProperties.ImgSize.height, TSDatasetProperties.ObjDatasetProperties.ImgSize.width);
		unsigned long label;

		batch_samples.clear();
		batch_labels.clear();

		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetSample(gray_img_buf, label);
			//Преобразуем ее в dlib::matrix
			CVMatToDlibMatrix8U(gray_img_buf, temp);
			resize_image(temp, sample);
			batch_samples.push_back(sample);
			batch_labels.push_back(label);
		};
	}
};

#endif