#ifndef TDNN_CLASSIFIER_H
#define TDNN_CLASSIFIER_H

#include <Resnet.h>
#include <dlib/gui_widgets.h>
#include <opencv2/core.hpp>

///Параметры обучения классификатора 
struct TDNNClassifierProperties {
	bool	UseDataAugmentation,
				SetDNNPreferSmallestAlgorithms;
	float StartLearningRate,
				MinLearningRate;
	size_t	MiniBatchSize;
	unsigned long SynchronizationPeriod,			//sec
								IterationsWithoutProgressThreshold;
	std::string SyncFile;
};

///Параметры обучения классификатора по умолчанию по умолчанию
static const TDNNClassifierProperties DNN_CLASSIFIER_PROPERTIES_DEFAULTS = 
{
	true,								//UseDataAugmentation
	true,								//SetDNNPreferSmallestAlgorithms
	0.01,								//StartLearningRate
	1e-6,								//MinLearningRate
	32,									//MiniBatchSize
	300,								//SynchronizationPeriod
	5000,								//IterationsWithoutProgressThreshold
	""									//SyncFile
};

///
template < typename TTrainingNet, typename TWorkingNet, typename TClassifierDataset >
class TDNNClassifier {
protected:
	TDNNClassifierProperties DNNClassifierProperties;
	TWorkingNet WorkingNet;
public:
	TDNNClassifier(const TDNNClassifierProperties &dnn_classifier_properties)
	{
		DNNClassifierProperties = dnn_classifier_properties;
	};

	virtual ~TDNNClassifier()
	{
	}

	///
	virtual TTrainingNet TrainCNN(TClassifierDataset * dataset, bool &success)
	{
		TTrainingNet result;
		success = false;
		dlib::image_window win1;

		std::cout<<"net initialization..."<<std::endl;
		//The code below uses mini-batch stochastic gradient descent with an initial learning rate of 0.01 to accomplish this.
		std::cout<<"trainer initialization..."<<std::endl;
		// dlib uses cuDNN under the covers.  One of the features of cuDNN is the
		// option to use slower methods that use less RAM or faster methods that use
		// a lot of RAM.  If you find that you run out of RAM on your graphics card
		// then you can call this function and we will request the slower but more
		// RAM frugal cuDNN algorithms.
		if (DNNClassifierProperties.SetDNNPreferSmallestAlgorithms)
			dlib::set_dnn_prefer_smallest_algorithms();

		//dlib::dnn_trainer<TTrainingNet> trainer(result);
		dlib::dnn_trainer<TTrainingNet, dlib::adam> trainer(result, dlib::adam(1.e-8, 0.9, 0.999)/*,{0,1}*/);
		trainer.set_learning_rate(DNNClassifierProperties.StartLearningRate);		//1e-3
		trainer.set_min_learning_rate(DNNClassifierProperties.MinLearningRate);
		trainer.set_mini_batch_size(DNNClassifierProperties.MiniBatchSize);
		trainer.be_verbose();
		trainer.set_synchronization_file(DNNClassifierProperties.SyncFile, std::chrono::seconds(DNNClassifierProperties.SynchronizationPeriod));
		trainer.set_iterations_without_progress_threshold(DNNClassifierProperties.IterationsWithoutProgressThreshold);
		std::cout<<"get_iterations_without_progress_threshold "<<trainer.get_iterations_without_progress_threshold()<<std::endl;
		//trainer.set_learning_rate_shrink_factor(0.1);
	
		//Обучение
		std::vector<dlib::matrix<unsigned char>> mini_batch_samples;
		std::vector<unsigned long> mini_batch_labels;
		dlib::rand rnd(time(0));
		std::cout<<"traning..."<<std::endl;
		const long double start_time = time(0);
		while (trainer.get_learning_rate() >= DNNClassifierProperties.MinLearningRate)
		{
			dataset->GetSampleBatch(mini_batch_samples, mini_batch_labels, DNNClassifierProperties.MiniBatchSize);

			win1.set_image(mini_batch_samples[0]);
			trainer.train_one_step(mini_batch_samples, mini_batch_labels);
		}
		// When you call train_one_step(), the trainer will do its processing in a
		// separate thread. However, this also means we need to wait for any mini-batches that are
		// still executing.  Calling get_net() performs the necessary synchronization.
		trainer.get_net();
		success = true;
		return result;
	}

	///
	virtual void Save(TTrainingNet * net, const std::string filename)
	{
		net->clean();																	//Очищаем сеть от вспомогательной информации
		dlib::serialize(filename) << *net;
	}
	///
	virtual void Load(const std::string filename)
	{
		dlib::deserialize(filename) >> WorkingNet;
	}

	///Предсказание 
	virtual void Predict(const dlib::matrix<unsigned char> &sample_dlib, unsigned long &result)
	{
		WorkingNet(sample_dlib);
		float max_response;
		for (unsigned long iv = 0; iv < dlib::layer<1>(WorkingNet).get_output().size(); iv++)
		{
			if (iv == 0 || dlib::layer<1>(WorkingNet).get_output().host()[iv] > max_response)
			{
				max_response = dlib::layer<1>(WorkingNet).get_output().host()[iv];
				result = iv;
			}
		}
	}

  ///Предсказание
	virtual void Predict(const cv::Mat &sample, unsigned long &result)
	{
		dlib::matrix<unsigned char> sample_dlib;
		CVMatToDlibMatrix8U(sample, sample_dlib);
		Predict(sample_dlib, result);
	}
};

#endif