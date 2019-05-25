#ifndef TTRAFFIC_SIGN_RECOGNIZER_H
#define TTRAFFIC_SIGN_RECOGNIZER_H

#include <TDNNClassifier.h>
#include <TTrafficSignDataset.h>

const unsigned long TRAFFIC_SIGN_NUMBER_OF_LEARNING_CLASSES = 44;
const unsigned long TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS = 3;

template <typename SUBNET> using res1 =	res<1, SUBNET>;
template <typename SUBNET> using ares1 =	ares<1, SUBNET>;
template <typename SUBNET> using res2 =	res<2, SUBNET>;
template <typename SUBNET> using ares2 =	ares<2, SUBNET>;
template <typename SUBNET> using res4 =	res<4, SUBNET>;
template <typename SUBNET> using ares4 =	ares<4, SUBNET>;
template <typename SUBNET> using res8 =	res<8, SUBNET>;
template <typename SUBNET> using ares8 =	ares<8, SUBNET>;
template <typename SUBNET> using res16 =	res<16, SUBNET>;
template <typename SUBNET> using ares16 =	ares<16, SUBNET>;
template <typename SUBNET> using res32 =	res<32, SUBNET>;
template <typename SUBNET> using ares32 =	ares<32, SUBNET>;
template <typename SUBNET> using res64 =	res<64, SUBNET>;
template <typename SUBNET> using ares64 =	ares<64, SUBNET>;
template <typename SUBNET> using res128 =	res<128, SUBNET>;
template <typename SUBNET> using ares128 =	ares<128, SUBNET>;
template <typename SUBNET> using res256 =	res<256, SUBNET>;
template <typename SUBNET> using ares256 =	ares<256, SUBNET>;

template <typename SUBNET> using level1 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,res256,res_down<256,SUBNET>>;
template <typename SUBNET> using level2 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,res128,res_down<128,SUBNET>>;
template <typename SUBNET> using level3 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,res64,res_down<64,SUBNET>>;
template <typename SUBNET> using level4 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,res32,res_down<32,SUBNET>>;

template <typename SUBNET> using alevel1 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,ares256,ares_down<256,SUBNET>>;
template <typename SUBNET> using alevel2 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,ares128,ares_down<128,SUBNET>>;
template <typename SUBNET> using alevel3 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,ares64,ares_down<64,SUBNET>>;
template <typename SUBNET> using alevel4 = dlib::repeat<TRAFFIC_SIGN_NUMBER_OF_INT_RES_LAYERS,ares32,ares_down<32,SUBNET>>;


// training network type
using TTSTrainedNet = dlib::loss_multiclass_log<
												dlib::fc<TRAFFIC_SIGN_NUMBER_OF_LEARNING_CLASSES,
													dlib::avg_pool_everything<		 //dlib::avg_pool_everything<		softmax_all
															res256<
															level1<
															res128<
															level2<
															res64<
															level3<
															res32<
															level4<
															dlib::repeat<2, res16,
															dlib::max_pool<2,2,1,1,dlib::relu<dlib::bn_con<dlib::con<16,5,5,1,1,
															dlib::input<dlib::matrix<unsigned char>>
															>>>>>>>>>>>>>
													>
											>>;

// testing network type (replaced batch normalization with fixed affine transforms)
using TTSNet = dlib::loss_multiclass_log<
								dlib::fc<TRAFFIC_SIGN_NUMBER_OF_LEARNING_CLASSES, 
									dlib::avg_pool_everything<						//dlib::avg_pool_everything<	 softmax_all
                            ares256<
														alevel1<
														ares128<
														alevel2<
														ares64<
														alevel3<
														ares32<
														alevel4<
														dlib::repeat<2, ares16,
                            dlib::max_pool<2,2,1,1,dlib::relu<dlib::affine<dlib::con<16,5,5,1,1,
                            dlib::input<dlib::matrix<unsigned char>>
                            >>>>>>>>>>>>>
									>
							>>;

///ѕараметры распознавани€ дорожных знаков
/*typedef*/ struct TTrafficSignClassifierProperties {
	TDNNClassifierProperties DNNClassifierProperties;
	TTrafficSignDatasetProperties TrafficSignDatasetProperties;
};
		
///ѕараметры распознавани€ дорожных знаков по умолчанию
static const TTrafficSignClassifierProperties TRAFFIC_SIGN_CLASSIFIER_PROPERTIES_DEFAULTS = 
{
	DNN_CLASSIFIER_PROPERTIES_DEFAULTS,
	TRAFFIC_SIGN_DATASET_PROPERTIES_DEFAULTS
};

///
class TTrafficSignClassifier : public TDNNClassifier <TTSTrainedNet, TTSNet, TTrafficSignDataset> {
protected:
	TTrafficSignClassifierProperties TrafficSignClassifierProperties;
public:
	TTrafficSignClassifier(const TTrafficSignClassifierProperties &traffic_sign_classifier_properties);
	virtual ~TTrafficSignClassifier();
};

#endif