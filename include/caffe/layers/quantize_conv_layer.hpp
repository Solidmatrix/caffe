#ifndef CAFFE_QUANTIZE_CONV_LAYER_HPP_
#define CAFFE_QUANTIZE_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/base_conv_layer.hpp"

namespace caffe {

#define InvokeOnce(x); \
static int need_invoke = 1; \
if (need_invoke) { \
  x; \
  need_invoke = 0; \
} \


template <typename Dtype>
class QuantizeConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
 
  explicit QuantizeConvolutionLayer(const LayerParameter& param)
      : BaseConvolutionLayer<Dtype>(param) {}

//   virtual ~QuantizeCuDNNConvolutionLayer();

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "QuantizeConvolution"; }

  virtual void QuantizeWeights(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

  
  static map<string, float> calibration_map_;
  float activation_scale_;
  float weight_scale_;
  int threshold_;
  Blob<Dtype> quantized_weight_Dtype_;
  Blob<Dtype> quantized_bias_Dtype_;
  Blob<Dtype> quantized_activation_Dtype_;
  bool need_quantize_;
};
template <typename Dtype>
map<string, float> QuantizeConvolutionLayer<Dtype>::calibration_map_;


}  // namespace caffe

#endif  // CAFFE_QUANTIZE_CONV_LAYER_HPP_
