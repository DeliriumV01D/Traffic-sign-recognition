#include <TTrafficSignClassifier.h>

TTrafficSignClassifier :: TTrafficSignClassifier(const TTrafficSignClassifierProperties &traffic_sign_classifier_properties)
	:	TDNNClassifier(traffic_sign_classifier_properties.DNNClassifierProperties)
{
	TrafficSignClassifierProperties = traffic_sign_classifier_properties;
}

TTrafficSignClassifier :: ~TTrafficSignClassifier()
{
}