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

///��������������� ��������� ��� �������� ������� ��������� ������� � �� �����
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

///��������� �������� ��� ������ � ��������� ��������
/*typedef*/ struct TDatasetProperties {
 	std::string Dir;
	cv::Size ImgSize;
	bool	UseMultiThreading,
				OneThreadReading,
				Warp;
};

///��������� �������� ��� ������ � ��������� �������� �� ���������
static const TDatasetProperties DATASET_PROPERTIES_DEFAULTS = 
{
	"",
	cv::Size(0, 0),
	true,
	false,
	false
};

///������� ��� ������ � ��������� ��������
class TDataset {
protected:
	TDatasetProperties DatasetProperties;
	TIndexedObjects <std::string> Labels;										///����� ������� - ����� ������������, ����� �� ������
	std::vector <TLabeledSamplePath> LabeledSamplePaths;		///������ ��������� ��������� ������� � ����� �������
	dlib::thread_pool * ThreadPool;
	std::mutex	ReadMutex,																	///�� ������ one_thread_reading ������� imread
							DetectMutex;
public:
	TDataset(const TDatasetProperties &dataset_properties);
	virtual ~TDataset();

	virtual unsigned long Size();
	virtual unsigned long ClassNumber();

	///�������� � �������������, ��� ��� ����������� ������ ������� ����� � ����� �����������
	virtual std::string GetLabelFromPath(const std::string &path);
	///�������� ������ �� ������� �� �������
	virtual cv::Mat GetSampleCVMat(const unsigned long sample_idx);
	///�������� ������ �� ������� �� �������
	virtual dlib::matrix<unsigned char> GetSampleDLibMatrix(const unsigned long sample_idx);
	///�������� ����� �� ������� �������
	virtual std::string GetLabel(const unsigned long sample_idx);
	///�������� ����� �� ������� �����
	virtual std::string GetLabelByIdx(const unsigned long label_idx);
	///�������� ������ ����� �� ������� �� �������
	virtual unsigned long GetLabelIdx(const unsigned long sample_idx);
	///������������ ���� ���������� �� ���� ������������(����������� �������!)
	virtual dlib::matrix<unsigned char> MakeInputSamplePair(dlib::matrix<unsigned char> * img1, dlib::matrix<unsigned char> * img2);

	///�������� ���� ����������� positive == true ������ �������, false - ������ �������� � ��������������� �����
	virtual void GetInputSamplePair(bool positive, dlib::matrix<unsigned char> &sample_pair, unsigned long &label);
	///�������� ����� ��� ����������� � ������������� �����
	virtual void GetInputSamplePairBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_sample_pairs, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);

	///��������� ������ �� �������� � ��������������� ��� �����	 cv::Mat - dlib::matrix ���������� ����� ������?
	virtual void GetRandomSample(dlib::matrix<unsigned char> &sample, unsigned long &label);
	///��������� ������ �� �������� � ��������������� ��� �����  cv::Mat - dlib::matrix ���������� ����� ������?
	virtual void GetRandomSample(cv::Mat &sample, unsigned long &label);
	///�������� ����� ����������� � ��������������� �� �����  cv::Mat - dlib::matrix ���������� ����� ������?
	virtual void GetRandomSampleBatch(
		std::vector<dlib::matrix<unsigned char>> &batch_samples, 
		std::vector<unsigned long> &batch_labels,
		const size_t batch_size
	);
	/////�������� ����� ����������� � ��������������� �� �����  cv::Mat - dlib::matrix ���������� ����� ������?
	//virtual void GetRandomSampleBatch(
	//	cv::Mat &data, 
	//	cv::Mat &classes,
	//	const size_t batch_size
	//);
};

#endif