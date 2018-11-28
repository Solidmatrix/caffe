#include <vector>

#include "caffe/layers/quantize_conv_layer.hpp"
#include <fstream>

namespace caffe {

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  BaseConvolutionLayer<Dtype>::LayerSetUp(bottom, top);

  // initialize calibration table
  if (this->calibration_map_.empty()) {
    string calibration_table = this->layer_param().calibration_param().calibration_table_name();
    CHECK_NE(calibration_table, "") << "[QuantizeConvolutionLayer<Dtype>::LayerSetUp] Calibration table is empty.";
    std::ifstream in(calibration_table.c_str());
    CHECK(in) << "[QuantizeConvolutionLayer<Dtype>::LayerSetUp] Calibration table file not found: " << calibration_table;
    LOG(INFO) << "[QuantizeConvolutionLayer<Dtype>::LayerSetUp] Calibration table file found:" << calibration_table;
    string name;
    float value;
    while (!in.eof()){
      in >> name >> value;
      this->calibration_map_[name] = value;
    }
    in.close();
  }
  // initialize quantization scales
  string conv_name = this->layer_param().name();
  activation_scale_ = this->calibration_map_[conv_name];  // activation scale
  weight_scale_ = this->calibration_map_[conv_name + "_param_0"];  // weight scale
  LOG(INFO) << "[QuantizeConvolutionLayer<Dtype>::LayerSetUp] scale of " << conv_name << ": " << activation_scale_ << ", " << weight_scale_;

  need_quantize_ = true;
  threshold_ = this->layer_param().calibration_param().threshold();
}

// quantize weight, invoked after weight file is read
template<typename Dtype>
void QuantizeConvolutionLayer<Dtype>::QuantizeWeights(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(INFO) << "[QuantizeConvolutionLayer<Dtype>::QuantizeWeights]";
  quantized_activation_Dtype_.Reshape(bottom[0]->shape());

  // quantize weight
  quantized_weight_Dtype_.Reshape(this->blobs_[0]->shape());
  const int count = this->blobs_[0]->count();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* quantized_weight = quantized_weight_Dtype_.mutable_cpu_data();
  for (int c = 0; c < count; ++c) {
    quantized_weight[c] = round(weight[c] * weight_scale_);
  }

  // quantize bias
  if (this->bias_term_) {
    quantized_bias_Dtype_.Reshape(this->blobs_[1]->shape());
    const int count = this->blobs_[1]->count();
    const Dtype* bias = this->blobs_[1]->cpu_data();
    Dtype* quantized_bias = quantized_bias_Dtype_.mutable_cpu_data();
    for (int c = 0; c < count; ++c) {
      quantized_bias[c] = round(bias[c] * weight_scale_ * activation_scale_);
      // need threshold here ?
    }
  }
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

// since cblas do not support int8 gemm,
// we use float gemm in cpu forwarding
template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InvokeOnce(LOG(INFO) << "[QuantizeConvolutionLayer<Dtype>::Forward_cpu]");
  
  if (need_quantize_) {
    QuantizeWeights(bottom, top);
    need_quantize_ = false;
  }

  const Dtype* quantized_weight = quantized_weight_Dtype_.cpu_data();
  Dtype* quantized_activation = quantized_activation_Dtype_.mutable_cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();

    // quantize activation
    const int count_bottom = bottom[i]->count();
    for (int c = 0; c < count_bottom; ++c) {
      Dtype quantized_value = round(bottom_data[c] * activation_scale_);
      quantized_activation[c] = (quantized_value > threshold_) ? threshold_ : ((quantized_value < -threshold_) ? -threshold_ : quantized_value);
    }

    // convolution
    for (int n = 0; n < this->num_; ++n) {
      this->forward_cpu_gemm(quantized_activation + n * this->bottom_dim_, quantized_weight,
          top_data + n * this->top_dim_);
    }

    // add bias
    for (int n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        const Dtype* bias = quantized_bias_Dtype_.cpu_data();
        this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
      }
    }

    // restore scale
    const int count_top = top[i]->count();
    for (int c = 0; c < count_top; ++c) {
      top_data[c] /= activation_scale_ * weight_scale_;
    }
  }
}















template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(0, 1) << "[QuantizeConvolutionLayer<Dtype>::Backward_cpu] do not support quantize backward";

  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(QuantizeConvolutionLayer);
#endif

INSTANTIATE_CLASS(QuantizeConvolutionLayer);
// REGISTER_LAYER_CLASS(QuantizeConvolution);

}  // namespace caffe
