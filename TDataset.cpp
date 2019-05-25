#include <TDataset.h>
#include <list>
#include <vector>
#include <time.h>
#include <WriteToLog.h>
#include <RandomWarpedImage.h>
#include <opencv2/highgui/highgui.hpp>
#include <dlib/matrix.h>
#include <dlib/threads.h>

const string LABEL_FILE_NAME = "label.txt";

//��������� ������ ����� ������ ����������� � ��������������� �� �����
TDataset :: TDataset(const TDatasetProperties &dataset_properties)
{
	DatasetProperties = dataset_properties;
	TRandomInt * ri = &(TRandomInt::Instance());
	ri->Initialize(time(0));

	unsigned long label;
	std::list <std::string> :: const_iterator	current_position;  
	std::list <std::string> data_list = getFilesNames(DatasetProperties.Dir, "*.bmp");
	data_list.splice(data_list.end(), getFilesNames(DatasetProperties.Dir, "*.jpg"));
	data_list.splice(data_list.end(), getFilesNames(DatasetProperties.Dir, "*.jpeg"));
	data_list.splice(data_list.end(), getFilesNames(DatasetProperties.Dir, "*.ppm"));

	for (	current_position = data_list.begin(); 
				current_position != data_list.end(); 
				++current_position
			)
	{
		label = Labels.GetIdxByObj(GetLabelFromPath(*current_position));
		LabeledSamplePaths.push_back(TLabeledSamplePath(*current_position, label));
	};
}

TDataset :: ~TDataset()
{}

//�������� � �������������, ��� ��� ����������� ������ ������� ����� � ����� �����������
std::string TDataset :: GetLabelFromPath(const std::string &path)
{
	std::string s;
	size_t l = path.find_last_of("/\\");
	if (ReadStringFromFile(path.substr(0, l + 1) + LABEL_FILE_NAME, s))
		return s;
	return path.substr(0, l + 1);
}

unsigned long TDataset :: Size()
{
	return LabeledSamplePaths.size();
}

unsigned long TDataset :: ClassNumber()
{
	return Labels.Size();
}

//�������� ������ �� ������� �� �������
cv::Mat TDataset :: GetSampleCVMat(const unsigned long sample_idx)
{
	if (sample_idx < 0 || sample_idx >= this->Size()) throw TException("Error TDataset :: GetSample: sample index is incorrect");

	cv::Mat result;
	//������ �������� ��������
	bool lock = DatasetProperties.UseMultiThreading && DatasetProperties.OneThreadReading;
	if (lock) ReadMutex.lock();
	try {
		result = cv::imread(LabeledSamplePaths[sample_idx].SamplePath.c_str(), CV_LOAD_IMAGE_GRAYSCALE);
	} catch (exception &e) {
		if (lock) ReadMutex.unlock();

		TException * E = dynamic_cast<TException *>(&e);
		if (E) throw (*E);
		else throw e;
	};
	if (lock) ReadMutex.unlock();
  if (!result.data) throw TException("Error TDataset :: GetSample: can not read image from path: " + LabeledSamplePaths[sample_idx].SamplePath);

	if (DatasetProperties.Warp && (RandomInt() % 3 > 0)) result = GetRandomWarpedImage(result, 0.15, 0.15, 0.2, true);
	return result;
}

//�������� ������ �� ������� �� �������
dlib::matrix<unsigned char> TDataset :: GetSampleDLibMatrix(const unsigned long sample_idx)
{
	dlib::matrix<unsigned char> temp, 
															result(DatasetProperties.ImgSize.height, DatasetProperties.ImgSize.width);
	cv::Mat gray_img_buf = GetSampleCVMat(sample_idx);
	//����������� �� � dlib::matrix
	CVMatToDlibMatrix8U(gray_img_buf, temp);
	resize_image(temp, result);
	return result;
}

//�������� ����� �� ������� �������
std::string TDataset :: GetLabel(const unsigned long sample_idx)
{
	return GetLabelByIdx(LabeledSamplePaths[sample_idx].Label);
}

///�������� ����� �� ������� �����
std::string TDataset :: GetLabelByIdx(const unsigned long label_idx)
{
	return Labels.GetObjByIdx(label_idx);
}

//�������� ������ ����� �� ������� �� �������
unsigned long TDataset :: GetLabelIdx(const unsigned long sample_idx)
{
	return LabeledSamplePaths[sample_idx].Label;
}

//������������ ���� ���������� �� ���� ������������(����������� �������!)
dlib::matrix<unsigned char> TDataset :: MakeInputSamplePair(dlib::matrix<unsigned char> * img1, dlib::matrix<unsigned char> * img2)
{
	dlib::matrix<unsigned char> result;
	result.set_size(img1->nr() * 2, img1->nc());
	dlib::set_subm(result, dlib::range(0, img1->nr() - 1), dlib::range(0, img1->nc() - 1)) = *img1;	//rectangle(0,0,1,2)
	dlib::set_subm(result, dlib::range(img2->nr(), img2->nr() * 2 - 1), dlib::range(0, img2->nc() - 1)) = *img2;	//rectangle(0,0,1,2)
	return result;
}

///�������� ���� ����������� positive == true ������ �������, false - ������ �������� � ��������������� �����
void TDataset :: GetInputSamplePair(bool positive, dlib::matrix<unsigned char> &sample_pair, unsigned long &label)
{
	unsigned long idx, jdx;
	dlib::matrix<unsigned char> img1, img2;
	const unsigned long locality = 30,
											attempts = 200;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//����� ��������� ������
		idx = RandomInt() % this->Size();
		if (positive){ //���� positive, �� ���� � �����������, ���� �� ����� ���������� label
			for (unsigned long i = 0; i < attempts; i++)
			{
				jdx = idx - locality/2 + RandomInt() % locality;
				if (jdx < 0 || jdx > this->Size() - 1) continue;
				if (GetLabelIdx(idx) == GetLabelIdx(jdx))
				{
					found = true;
					break;
				};
			}
		} else {//if positive
			//���� negative, �� ������ ��������� ����� ���������� �������, ���� �� ��������� �� label, ������������
			for (unsigned long i = 0; i < attempts; i++)
			{
				jdx = RandomInt() % this->Size();
				if (GetLabelIdx(idx) != GetLabelIdx(jdx)) 
				{
					found = true;
					break;
				};
			}
		};
		try {
			img1 = GetSampleDLibMatrix(idx);
			img2 = GetSampleDLibMatrix(jdx);
		} catch (exception &e) {
			found = false;
			TException * E = dynamic_cast<TException *>(&e);
			if (E)
				std::cout<<E->what()<<std::endl;
			else
				std::cout<<e.what()<<std::endl;
		};
	} while (!found);
	sample_pair = MakeInputSamplePair(&img1, &img2);
	if (positive) label = 1; else label = 0;
}

struct TR {
	TDataset * Dataset; 
	bool Positive; 
	dlib::matrix<unsigned char> SamplePair;
	unsigned long Label;
};

//�������� ����� ��� ����������� � ������������� �����
void TDataset :: GetInputSamplePairBatch(
	std::vector<dlib::matrix<unsigned char>> &batch_sample_pairs, 
	std::vector<unsigned long> &batch_labels,
	const size_t batch_size
){
	bool positive = true;
	dlib::matrix<unsigned char> sample_pair;
	unsigned long label;

	batch_sample_pairs.clear();
	batch_labels.clear();
	if (DatasetProperties.UseMultiThreading)
	{
		std::vector <TR> trv;
		std::vector<dlib::future<TR>> fv(batch_size);
		for (unsigned long i = 0; i < batch_size; i++)
		{
			trv.push_back(TR() = {this, positive});
	    fv[i] = trv[i];
			//�����, ������� �� �������� � ��� ������ �����
			ThreadPool->add_task_by_value([](TR &val){val.Dataset->GetInputSamplePair(val.Positive, val.SamplePair, val.Label);}, fv[i]);
			positive = !positive;	//�������� ������������� � ������������� �������
		};
		//������� ������ ����� ���� ��������� ���������, ������ ��� ����� ���� ������� ������� � ��.
		ThreadPool->wait_for_all_tasks();
		for (unsigned long i = 0; i < batch_size; i++)
		{
			batch_sample_pairs.push_back(fv[i].get().SamplePair);
			batch_labels.push_back(fv[i].get().Label);
			//batch_sample_pairs.push_back(tfv[i].SamplePair);
			//batch_labels.push_back(tfv[i].Label);
		};
	} else {
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetInputSamplePair(positive, sample_pair, label);
			batch_sample_pairs.push_back(sample_pair);
			batch_labels.push_back(label);
			positive = !positive;	//�������� ������������� � ������������� �������
		};
	}
}

//��������� ������ �� �������� � ��������������� ��� �����
void TDataset :: GetRandomSample(dlib::matrix<unsigned char> &sample, unsigned long &label)
{
	unsigned long idx;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//����� ��������� ������
		idx = RandomInt() % this->Size();
		try {
			sample = GetSampleDLibMatrix(idx);
			label = GetLabelIdx(idx);
			found = true;
		} catch (exception &e) {
			found = false;
			//TException * E = dynamic_cast<TException *>(&e);
			//if (E)
			//	std::cout<<E->what()<<std::endl;
			//else
			//	std::cout<<e.what()<<std::endl;
		};
	} while (!found);
}

//��������� ������ �� �������� � ��������������� ��� �����
void TDataset :: GetRandomSample(cv::Mat &sample, unsigned long &label)
{
	unsigned long idx;
	if (this->Size() == 0) return;
	bool found = false;
	
	do {
		//����� ��������� ������
		idx = RandomInt() % this->Size();
		try {
			sample = GetSampleCVMat(idx);
			label = GetLabelIdx(idx);
			found = true;
		} catch (exception &e) {
			found = false;
		};
	} while (!found);
}

struct TS {
	TDataset * Dataset; 
	dlib::matrix<unsigned char> Sample;
	unsigned long Label;
};

//�������� ����� ����������� � ��������������� �� �����  cv::Mat - dlib::matrix ���������� ����� ������?
void TDataset :: GetRandomSampleBatch(
	std::vector<dlib::matrix<unsigned char>> &batch_samples, 
	std::vector<unsigned long> &batch_labels,
	const size_t batch_size
) {
	dlib::matrix<unsigned char> sample;
	unsigned long label;

	batch_samples.clear();
	batch_labels.clear();
	if (DatasetProperties.UseMultiThreading)
	{
		std::vector <TS> tsv;
		std::vector<dlib::future<TS>> fv(batch_size);
		for (unsigned long i = 0; i < batch_size; i++)
		{
			tsv.push_back(TS() = {this});
	    fv[i] = tsv[i];
			//�����, ������� �� �������� � ��� ������ �����
			ThreadPool->add_task_by_value([](TS &val){val.Dataset->GetRandomSample(val.Sample, val.Label);}, fv[i]);
		};
		//������� ������ ����� ���� ��������� ���������, ������ ��� ����� ���� ������� ������� � ��.
		ThreadPool->wait_for_all_tasks();
		for (unsigned long i = 0; i < batch_size; i++)
		{
			batch_samples.push_back(fv[i].get().Sample);
			batch_labels.push_back(fv[i].get().Label);
		};
	} else {
		for (unsigned long i = 0; i < batch_size; i++)
		{
			GetRandomSample(sample, label);
			batch_samples.push_back(sample);
			batch_labels.push_back(label);
		};
	}
}