//#include <image_utils.h>
#include <cascade_detector.h>


BaseCascadeDetector :: BaseCascadeDetector(const cv::Size &min_size)
{
	MinSize = min_size;
};

BaseCascadeDetector :: ~BaseCascadeDetector()
{
  for (unsigned int i = 0; i < classifiers.size(); i++)
    delete (classifiers[i]);

  for (unsigned int i = 0; i < descriptors.size(); i++)
    delete (descriptors[i]);
};

void BaseCascadeDetector :: addStep(
  TBaseDescriptor * descriptor,
  TBaseClassifier * classifier
){
  descriptors.push_back(descriptor);
  classifiers.push_back(classifier);
};


//Обучаем каскад классификаторов на данных для объектов и фона. 
void BaseCascadeDetector :: train (
  const char * objects_train_data_path,
  const char * background_train_data_path,
  float &fp, 
  float &tn
) {
  static const int n_test = 250;
  int i, j, delta_i;
  unsigned int m;

  cv::Mat positive,
          negative,
          train_data,
          train_classes,
          test_data,
          test_classes;

  for (m = 0; m < classifiers.size(); m++)
  {
    //cout<<positive.type()<<" "<<Descriptors[m]->GetTrainingVectorFromImageCollection(objects_train_data_path).type()<<endl;
    positive = descriptors[m]->GetTrainingVectorFromImageCollection(objects_train_data_path);
    negative = descriptors[m]->GetTrainingVectorFromImageCollection(background_train_data_path);
    train_data = cv::Mat::zeros(positive.rows + negative.rows - n_test*2, positive.cols, CV_32FC1);
    train_classes = cv::Mat::zeros(positive.rows + negative.rows - n_test*2, 1, CV_32FC1);
    test_data = cv::Mat::zeros(n_test*2, positive.cols, CV_32FC1);
    test_classes = cv::Mat::zeros(n_test*2, 1, CV_32FC1);

    //Собираем данные в одной матрице обучения trainData и определяем принадлежность данных к классу в trainClasses
    //cvCopy(positive, trainData);
    delta_i = 0;
    for (i = 0; i < positive.rows + negative.rows; i++)
      for (j = 0; j < positive.cols; j++)
      {
        if (i < positive.rows)
        {
          if ((i % int(positive.rows/n_test) == 0) &&
          (i != 0) &&
          (delta_i < n_test)
          ){
            //cout<<"+"<<delta_i<<" "<<i<<" "<<positive.rows<<endl;
            test_data.at<float>(delta_i, j) = positive.at<float>(i, j);
            test_classes.at<float>(delta_i) = 1.;
            if (j == positive.cols - 1) delta_i += 1;
          } else {
            train_data.at<float>(i - delta_i, j) = positive.at<float>(i, j);
            train_classes.at<float>(i - delta_i) = 1.;
          }
        } else {
          if (
            ((i - positive.rows) % int(negative.rows/n_test) == 0) &&
            ((i - positive.rows) / int(negative.rows/n_test) > 1) &&
            (delta_i < 2 * n_test)
          ){
            //cout<<"-"<<delta_i<<" "<<i<<" "<<i - positive.rows<<" "<<negative.rows<<endl;
            test_data.at<float>(delta_i, j) = negative.at<float>(i - positive.rows, j);
            test_classes.at<float>(delta_i) = 0.;
            if (j == positive.cols - 1) delta_i += 1;
          } else {
            if (i < positive.rows + negative.rows - n_test*2)
            {
              train_data.at<float>(i - delta_i, j) = negative.at<float>(i - positive.rows, j);
              train_classes.at<float>(i - delta_i) = 0.;
            };
          }
        };
      };
    classifiers[m]->Train(train_data, train_classes);
  };
	
  //Выделив, предварительно, часть выборки в тестовую выборку, проводим на ней тестирование
  fp = tn = 0.;
  cv::Mat temp_mat(1, test_data.cols, test_data.type()),
          temp_result(1,1, CV_32FC1);
  float temp_float;
  for (i = 0; i < test_data.rows; i++)
  {
    //for (j = 0; j < test_data.cols; j++)
    //	temp_mat.at<float>(j) = test_data.at<float>(i, j);
    temp_mat = test_data(cv::Rect(0, i, test_data.cols, 1));
    classifiers[classifiers.size() - 1]->Predict(temp_mat, temp_result);
    temp_float = temp_result.at<float>(0, 0);
    if (test_classes.at<float>(i) != temp_float)
    {
      if (test_classes.at<float>(i) == 1.) fp += 1.;
      if (test_classes.at<float>(i) == 0.) tn += 1.;
    };
  }
  fp/=n_test;
  tn/=n_test;
};

//Сохранение параметров классификаторов
void BaseCascadeDetector :: save()
{
  std::stringstream ss;
  for (unsigned int i = 0; i < classifiers.size(); i++)
  {
    ss.str("");
    ss<<"classifier_"<<i<<".dat";
    classifiers[i]->Save(ss.str());
  };
};

//Загрузка параметров классификаторов
void BaseCascadeDetector :: load()
{
  std::stringstream ss;
  for (unsigned int i = 0; i < classifiers.size(); i++)
  {
    ss.str("");
    ss<<"classifier_"<<i<<".dat";
    classifiers[i]->Load(ss.str());
  };
};

//Классифицировать каскадом классификаторов
float BaseCascadeDetector :: predict(const cv::Mat &mat)
{
  cv::Mat mat_result(1, 1, CV_32FC1),
          temp_tdd;
  float result = 0;
  //for (unsigned int i = 0; i < classifiers.size(); i++)
  for (unsigned int i = 0; i < 2; i++)
  {
    //Вычисляем дескриптор от фрагмента внутри прямоугольника
    temp_tdd = cv::Mat(1, descriptors[i]->GetDescriptorSize(), CV_32FC1);
    descriptors[i]->CalculateImageDescriptor(
      &mat,
      &temp_tdd
    );
    classifiers[i]->Predict(temp_tdd, mat_result);
    result = mat_result.at<float>(0, 0);
    if (result <= 0) break;
  };
  return result;
};

//Детектирование скользящим окном
void BaseCascadeDetector :: detectMultiscale(
  const cv::Mat &image,
  std::vector<cv::Rect> &found_locations
) {
  unsigned int n, nw, nh, i, j, k, m;
  cv::Rect temp_rect;
  cv::Mat temp_tdd;
  cv::Mat resized_img_buf,
          gray_img_buf;
  std::vector <cv::Mat> tdd_vector;

  found_locations.clear();
  
  //Переводим в градации серого
  if (image.type() != CV_8UC1)
  {
    cv::cvtColor(image, gray_img_buf, CV_BGR2GRAY);
    gray_img_buf.convertTo(gray_img_buf, CV_8UC1);
  } else {
    image.copyTo(gray_img_buf);
  };
  //!!!В предположении, что все дескрипторы каскада настроены на один размер окна DescriptedWindowSize
  resized_img_buf = cv::Mat(descriptors[0]->GetDescriptedWindowSize(), CV_8UC1);

  for (m = 0; m < descriptors.size(); m++)
  {
    if (descriptors[m]->IsGlobalDescriptor()) descriptors[m]->PrepareMultiscale(&gray_img_buf);
    temp_tdd = cv::Mat(1, descriptors[m]->GetDescriptorSize(), CV_32FC1);
    tdd_vector.push_back(temp_tdd);
  };

  //Находим максимальное увеличение окна детектора(кратно размеру окна дескриптора)
  nw = image.cols / MinSize.width;
  nh = image.rows / MinSize.height;
  if (nw < nh) n = nw; else n = nh;

  //Mat TEMP_IMG;
  //image.copyTo(TEMP_IMG);

  //Метод скользящего окна с перекрытием на одну четверть/треть
  for (i = 0; i < n; i++) //цикл по масштабам окна
  {
    if (i == 0)
    {
      temp_rect.width = (int)((double)MinSize.width * 1.25);				//1.25; 1.5;
      temp_rect.height = (int)((double)MinSize.height * 1.25);			//1.25; 1.5;
		} else {
      temp_rect.width = MinSize.width * i;
      temp_rect.height = MinSize.height * i;
    }
    nw = int(4 * image.size().width / (temp_rect.width)) - 3;			//4 -3//3 -2
    nh = int(4 * image.size().height / (temp_rect.height)) - 3;		//4 -3//3 -2
    if ((nw < 1) || (nh < 1)) continue;

    cv::Mat temp_mat(temp_rect.size()/*cvGetSize(&image)*/, CV_8UC1);
    float res;
    cv::Mat mat_res(1, 1, CV_32FC1);

    //Циклы по вертикальным и горизонтальным позициям
    for (j = 0; j < nw; j++)
      for (k = 0; k < nh; k++)
      {
        temp_rect.x = j * int(temp_rect.width/4);				// /4, /3
        temp_rect.y = k * int(temp_rect.height/4);      // /4, /3

        temp_mat = gray_img_buf(temp_rect);

        //Для все ступеней каскада дескрипторов и классификаторов
        for (m = 0; m < descriptors.size(); m++)
        {
          if (descriptors[m]->IsGlobalDescriptor()) {
            //Вычисление дескриптора от области предподготовленной картинки
            descriptors[m]->CalculateImageDescriptor(
							&temp_rect,
							&tdd_vector[m]
            );
          } else {
            //Вычисляем дескриптор от фрагмента внутри прямоугольника
            descriptors[m]->CalculateImageDescriptor(
              &temp_mat,
              &tdd_vector[m],
              &resized_img_buf
            );
          }

          //Классифицируем фрагмент
          classifiers[m]->Predict(tdd_vector[m], mat_res);
          res = mat_res.at<float>(0, 0);
          ////Если нет признаков объекта, то классификаторы более высокого уровня не применяются
          //if (res <= 0) break;
											
					//Если нет признаков объекта, то классификаторы более высокого уровня не применяются
					if (res == 0) break;
					//Если на последнем этапе видим признаки объекта - добавляем область в дальнейшее рассмотрение
					if ((m == descriptors.size() - 1) && (res == 1))
						found_locations.push_back(temp_rect);
        };  //Для все ступеней каскада дескрипторов и классификаторов

        ////Если на последнем этапе видим признаки объекта - добавляем область в дальнейшее рассмотрение
        //if (res >= 0.5)
        //  found_locations.push_back(temp_rect);
      };    //Циклы по вертикальным и горизонтальным позициям
  };        //цикл по масштабам окна
};
