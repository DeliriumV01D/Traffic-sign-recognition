#ifndef CASCADE_DETECTOR_H
#define CASCADE_DETECTOR_H

#include <classifiers.h>
#include <descriptors.h>
#include <vector>

///Каскадный детектор (имеет несколько ступеней соответвующих друг другу дескрипторов и классификаторов)
class BaseCascadeDetector {
protected:
	cv::Size MinSize;
  std::vector <TBaseDescriptor *> descriptors;
  std::vector <TBaseClassifier *> classifiers;
public:
  BaseCascadeDetector(const cv::Size &min_size);
  virtual ~BaseCascadeDetector();

  ///Добавить ступень (дескриптор - классификатор)
  void addStep(TBaseDescriptor * descriptor, TBaseClassifier  * classifier);

  ///Обучаем каскад классификаторов на данных для объектов и фона. 
  void train (
    const char * objects_train_data_path,
    const char * background_train_data_path,
    float &fp, 
    float &tn
  );

  ///Сохранение параметров классификаторов
  void save();
  
  ///Загрузка параметров классификаторов
  void load();

  ///Классифицировать каскадом классификаторов
  float predict(const cv::Mat &mat);

  ///Детектирование скользящим окном
  void detectMultiscale(const cv::Mat &image, std::vector<cv::Rect> &found_locations);
};

#endif
