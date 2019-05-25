#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/highgui.hpp>
#include <TTrafficSignDataset.h>
#include <TTrafficSignClassifier.h>

#include <dlib/gui_widgets.h>

static const std::string	TRAFFIC_SIGN_RECOGNITION_OBJ_DATA_DIR = "Data/objects",
													TRAFFIC_SIGN_RECOGNITION_BACK_DATA_DIR = "Data/background",
													TRAFFIC_SIGN_RECOGNITION_TEST_OBJ_DATA_DIR = "Data/test_objects",
													TRAFFIC_SIGN_RECOGNITION_TEST_BACK_DATA_DIR = "Data/test_background",
													NET_DATA_FILE = "resnet_traffic_sign.dat",
													NET_SYNC_FILE = "resnet_traffic_sign_sync";
static const cv::Size IMAGE_SIZE = cv::Size(42, 42);

int main(int argc, char *argv[])
{
	setlocale(LC_ALL, "Russian");
	char c;

	if (argc > 1)
	{
		
		try {
			std::string ts = argv[1];
			cv::Mat tm,
							temp_mat;
			unsigned long label;
			//Чтение файла изображения
			tm = cv::imread(ts);

			//Инициализация классификатора дорожных знаков
			std::cout<<"loading classifier configuration..."<<std::endl;
			TTrafficSignClassifier traffic_sign_classifier(TRAFFIC_SIGN_CLASSIFIER_PROPERTIES_DEFAULTS);
			traffic_sign_classifier.Load(NET_DATA_FILE);
		
			//Методом скользящего окна проходим классификатором дорожных знаков по всем фрагментам изображения разных масштабов 
			std::cout<<"processing image..."<<std::endl;			
			MultiscaleSampler multi_sampler(IMAGE_SIZE);
			std::vector <cv::Rect> locations;
			multi_sampler.getSamples(tm, std::vector <cv::Rect>(), locations);
			for (size_t i = 0; i < locations.size(); i++)
			{
				std::cout<<i<<" of "<<locations.size()<<std::endl;
				ToGrayscale(tm(locations[i]), temp_mat);
				cv::resize(temp_mat, temp_mat, IMAGE_SIZE, 0, 0, CV_INTER_LINEAR);
				traffic_sign_classifier.Predict(temp_mat, label);
				if (label != TRAFFIC_SIGN_NUMBER_OF_LEARNING_CLASSES - 1)
				{
					//Вывод
					cv::imwrite("out_" + std::to_string(i) + ".jpg", tm(locations[i]));
					std::ofstream file;
					file.open ("out_" + std::to_string(i)+ ".txt");
					file << std::to_string(label);
					file.close();
					rectangle(tm, locations[i], cv::Scalar(100, 255, 100));
				}
			}		 			

			if (argc > 2)
			{
				cv::imshow("tw", tm);
				cv::waitKey(0);
			}
		} catch (exception &e) {
			TException * E = dynamic_cast<TException *>(&e);
			if (E)
				std::cerr<<E->what()<<std::endl;
			else
				std::cerr<<e.what()<<std::endl;
		};
		return 0;
	}

  std::cout<<"Choose operation:"<<std::endl
  <<"1. Learn traffic sign recognition algorithm"<<std::endl
  <<"2. Test traffic sign recognition algorithm"<<std::endl
	<<"any other char to exit"<<std::endl;
	std::cin>>c;

	if (c == '1')	//Learn traffic sign recognition algorithm
	{
		dlib::image_window win;

		//Инициализация обучающей выборки
		std::cout<<"dataset initialization..."<<std::endl;
		TTrafficSignDatasetProperties p = TRAFFIC_SIGN_DATASET_PROPERTIES_DEFAULTS;
		p.ObjDatasetProperties.Dir = TRAFFIC_SIGN_RECOGNITION_OBJ_DATA_DIR;
		p.ObjDatasetProperties.ImgSize = IMAGE_SIZE;
		p.ObjDatasetProperties.Warp = true;
		p.BackDatasetProperties.Dir = TRAFFIC_SIGN_RECOGNITION_BACK_DATA_DIR;
		p.BackDatasetProperties.ImgSize = IMAGE_SIZE;
		p.BackDatasetProperties.Warp = true;
		p.Ratio = 0.5;
		p.BackLabel = 43;
		
		TTrafficSignDataset dataset(p);

		//Training
		std::cout<<"training..."<<std::endl;
		bool result;
		TTrafficSignClassifierProperties tsc_properties = TRAFFIC_SIGN_CLASSIFIER_PROPERTIES_DEFAULTS;
		tsc_properties.TrafficSignDatasetProperties = p;
		tsc_properties.DNNClassifierProperties.SyncFile = NET_SYNC_FILE;
		tsc_properties.DNNClassifierProperties.MinLearningRate = 1e-6;
		TTrafficSignClassifier traffic_sign_classifier(tsc_properties);
				

		TTSTrainedNet net = traffic_sign_classifier.TrainCNN(&dataset, result);
		std::cout<<"train result: "<<result<<std::endl;
		if (result)
		{
			std::cout<<"saving/loading configuration..."<<std::endl;
			traffic_sign_classifier.Save(&net, NET_DATA_FILE);
			char c;
			std::cin>>c;
		}

		//for (unsigned long i = 0; i < 1000; i++)
		//{
		//	std::vector<dlib::matrix<unsigned char>> batch_samples; 
		//	std::vector<unsigned long> batch_labels;
		//	const size_t batch_size	= 64;
		//	dataset.GetSampleBatch(batch_samples, batch_labels, batch_size);
		//	win.set_image(batch_samples[63]);
		//	std::cout<<batch_labels[63]<<std::endl;
		//}
	}

	if (c == '2')	//Test traffic sign recognition algorithm
	{
		dlib::image_window win;
		int TP = 0,
		FP = 0,
		FN = 0,
		TN = 0,
		TOT = 10000;
		
		//Инициализация тестовой выборки
		std::cout<<"dataset initialization..."<<std::endl;
		TTrafficSignDatasetProperties p = TRAFFIC_SIGN_DATASET_PROPERTIES_DEFAULTS;
		p.ObjDatasetProperties.Dir = TRAFFIC_SIGN_RECOGNITION_TEST_OBJ_DATA_DIR;
		p.ObjDatasetProperties.ImgSize = IMAGE_SIZE;
		p.ObjDatasetProperties.Warp = false;
		p.BackDatasetProperties.Dir = TRAFFIC_SIGN_RECOGNITION_TEST_BACK_DATA_DIR;
		p.BackDatasetProperties.ImgSize = IMAGE_SIZE;
		p.BackDatasetProperties.Warp = false;
		p.Ratio = 0.1;
		p.BackLabel = 43;
		
		TTrafficSignDataset dataset(p);

		//Инициализация классификатора
		std::cout<<"loading classifier configuration..."<<std::endl;
		TTrafficSignClassifierProperties tsc_properties = TRAFFIC_SIGN_CLASSIFIER_PROPERTIES_DEFAULTS;
		tsc_properties.TrafficSignDatasetProperties = p;
		tsc_properties.DNNClassifierProperties.SyncFile = NET_SYNC_FILE;
		tsc_properties.DNNClassifierProperties.MinLearningRate = 1e-6;
		TTrafficSignClassifier traffic_sign_classifier(tsc_properties);
		traffic_sign_classifier.Load(NET_DATA_FILE);

		//Тестирование классификатора
		for (unsigned long i = 0; i < TOT; i++)
		{
			std::vector<dlib::matrix<unsigned char>> samples;
			std::vector<unsigned long> true_labels;
			unsigned long predicted_label;
			dataset.GetSampleBatch(samples, true_labels, 1);
			traffic_sign_classifier.Predict(samples[0], predicted_label);

			if (true_labels[0] != TRAFFIC_SIGN_NUMBER_OF_LEARNING_CLASSES - 1)
			{
				if (true_labels[0] == predicted_label)
					TP++;
				else
					FP++;
			}	else {
				if (true_labels[0] == predicted_label)
					FN++;
				else
					TN++;
			}

			win.set_image(samples[0]);
			std::cout<< i <<" of "<< TOT <<" predicted: "<<predicted_label<<"; ground truth: "<<true_labels[0]<<std::endl;
			cv::waitKey(1);
		}
		std::cout << "true positive = " << TP << ", false positives = " << FP << ", false negatives = " << FN << ", true negatives = " << TN << std::endl;
		std::cout << "Precision: " << (double)TP / (double)(TP + FP) << ", Recall: " << (double)TP / (double)(TP + FN) << std::endl;
	}
	
	std::cin>>c;
}