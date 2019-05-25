#ifndef MULTISCALE_SAMPLER_H
#define MULTISCALE_SAMPLER_H

#include <cascade_detector.h>

///Вспомогательный дескриптор к классу MultiscaleSampler
class MultiSamplerDescriptor : public TBaseDescriptor {
protected:
public:
  MultiSamplerDescriptor(const cv::Size &min_size) : TBaseDescriptor(TBaseDescriptorParams() = {min_size})
  {
    GlobalDescriptior = true;
  };
  ~MultiSamplerDescriptor(){};
  ///Размер (длина) дескриптора 
  size_t GetDescriptorSize(){return 4;};
  ///Размер окна/изображения, описываемого дескриптором без предварительных преобразований
  cv::Size GetDescriptedWindowSize(){return BaseParams.descripted_window_size;};
  ///Не используется
  void Compute(const cv::Mat * image, cv::Mat * descriptors){};
  ///Функция вычисления локального дескриптора
  /**Возвращает обратно параметры прямоугольника*/
  void Compute(const cv::Rect * rect, cv::Mat * descriptors)
  {
    descriptors->resize(1,4);
    descriptors->at<float>(0,0) = float(rect->x);
    descriptors->at<float>(0,1) = float(rect->y);
    descriptors->at<float>(0,2) = float(rect->width);
    descriptors->at<float>(0,3) = float(rect->height);
  };
  ///Предподготовка вычислений
  void PrepareMultiscale(cv::Mat * full_image){};
};

///Вспомогательный классификатор к классу TMultiscaleSampler
class MultiSamplerClassifier : public TBaseClassifier {
protected:
  std::vector<cv::Rect> zones;
public:
  MultiSamplerClassifier(){};
  ///Обучение 
  void Train(const cv::Mat &train_data, const cv::Mat &train_classes){};
  ///Сохранение параметров
  void Save(const std::string &file_name){};
  ///Загрузка параметров
  void Load(const std::string &file_name){};

  ///Устанавливает зоны, откуда выборка не берется
  void SetZones(const std::vector <cv::Rect> &zones)
  {
    this->zones = zones;
  };

  ///Классификация
  ///!!!Сейчас по-быстрому сделал под один дескриптор, хотя именно эта функция заточена под вектор дескрипторов
  float Predict(const cv::Mat &samples, cv::Mat &results)
  {
    results.at<float>(0,0) = Predict(samples);
		return 0;
  };

  ///Классификация
  float Predict(const cv::Mat &mat)
  {
    bool is_inside;
    const int x1 = int(mat.at<float>(0,0)),
    x2 = int(mat.at<float>(0,0) + mat.at<float>(0,2)),
    y1 = int(mat.at<float>(0,1)),
    y2 = int(mat.at<float>(0,1) + mat.at<float>(0,3));

    is_inside = false;
    for (unsigned int i = 0; i < zones.size(); i++)
    {
      if (
        cv::Point(x1, y1).inside(zones[i]) ||
        cv::Point(x2, y1).inside(zones[i]) ||
        cv::Point(x1, y2).inside(zones[i]) ||
        cv::Point(x2, y2).inside(zones[i])
      ) {
        is_inside = true;
        break;
      };
    }
    if (is_inside) return float(0);
    else return float(1);
  };
};

///Класс для кратномасштабного извлечения примеров выборки на основе изображения с выделенными запрещающими зонами
class MultiscaleSampler {
protected:
  BaseCascadeDetector * sampler;
  MultiSamplerClassifier * multi_sampler_classifier;
public:
  MultiscaleSampler(const cv::Size &min_size);
  ~MultiscaleSampler();

  void getSamples(
    const cv::Mat &frame,
    const std::vector <cv::Rect> &zones,
    std::vector <cv::Rect> &locations
  );
};

#endif
