#include <vector>

#include "caffe/layers/quantize_conv_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void QuantizeWithScaleAndThreshold(const int n, const Dtype* in, Dtype* out, const float scale, const int threshold) {
  CUDA_KERNEL_LOOP(index, n) {
    Dtype quantized_value = round(in[index] * scale);
    out[index] = (quantized_value > threshold) ? threshold : ((quantized_value < -threshold) ? -threshold : quantized_value);
  }
}
 
template <typename Dtype>
__global__ void RestoreScale(const int n, Dtype* out, const float scale) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = out[index] / scale;
  }
}

// since cublas do not support int8 gemm,
// we use float gemm in cpu forwarding
template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InvokeOnce(LOG(INFO) << "[QuantizeConvolutionLayer<Dtype>::Forward_gpu]");
  if (need_quantize_) {
    QuantizeWeights(bottom, top);
    need_quantize_ = false;
  }

  const Dtype* quantized_weight = quantized_weight_Dtype_.gpu_data();
  Dtype* quantized_activation = quantized_activation_Dtype_.mutable_gpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->gpu_data();
    Dtype* top_data = top[i]->mutable_gpu_data();

    // quantize activation
    const int count_bottom = bottom[i]->count();
    QuantizeWithScaleAndThreshold<Dtype><<<CAFFE_GET_BLOCKS(count_bottom), CAFFE_CUDA_NUM_THREADS>>>(
        count_bottom, bottom_data, quantized_activation, activation_scale_, threshold_);

    // convolution
    for (int n = 0; n < this->num_; ++n) {
      this->forward_gpu_gemm(quantized_activation + n * this->bottom_dim_, quantized_weight,
          top_data + n * this->top_dim_);
    }

    // add bias
    for (int n = 0; n < this->num_; ++n) {
      if (this->bias_term_) {
        const Dtype* quantized_bias = quantized_bias_Dtype_.gpu_data();
        this->forward_gpu_bias(top_data + n * this->top_dim_, quantized_bias);
      }
    }

    // restore scale
    const int count_top = top[i]->count();
    RestoreScale<Dtype><<<CAFFE_GET_BLOCKS(count_top), CAFFE_CUDA_NUM_THREADS>>>(
        count_top, top_data, activation_scale_ * weight_scale_);
  }
}

template <typename Dtype>
void QuantizeConvolutionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  CHECK_EQ(0, 1) << "[QuantizeConvolutionLayer<Dtype>::Backward_gpu] do not support quantize backward";

  const Dtype* weight = this->blobs_[0]->gpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->gpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      const Dtype* bottom_data = bottom[i]->gpu_data();
      Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        // gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
              top_diff + n * this->top_dim_, weight_diff);
        }
        // gradient w.r.t. bottom data, if necessary.
        if (propagate_down[i]) {
          this->backward_gpu_gemm(top_diff + n * this->top_dim_, weight,
              bottom_diff + n * this->bottom_dim_);
        }
      }
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(QuantizeConvolutionLayer);

}  // namespace caffe
