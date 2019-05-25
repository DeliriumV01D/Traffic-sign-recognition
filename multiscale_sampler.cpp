#include <multiscale_sampler.h>

/************************************************/
//MultiscaleSampler
/************************************************/ 

MultiscaleSampler :: MultiscaleSampler(const cv::Size &min_size)
{
	sampler = new BaseCascadeDetector(min_size);
  multi_sampler_classifier = new MultiSamplerClassifier();
  sampler->addStep(new MultiSamplerDescriptor(min_size), multi_sampler_classifier);
};

MultiscaleSampler :: ~MultiscaleSampler()
{
	delete sampler;
};

void MultiscaleSampler :: getSamples(
  const cv::Mat &frame,
  const std::vector <cv::Rect> &zones,
  std::vector <cv::Rect> &locations
){
  locations.clear();
  multi_sampler_classifier->SetZones(zones);
  sampler->detectMultiscale(frame, locations);
};
