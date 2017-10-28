#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/upsample_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
__global__ void UpsampleForward(const int nthreads, int in_w, int in_h,
    int out_w, int out_h, const Ftype* bottom_data,
    const Ftype* bottom_mask, Ftype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int offset = index / (in_w * in_h) * out_w * out_h;
    int upsample_idx = static_cast<int>(bottom_mask[index]);
    top_data[offset + upsample_idx] = bottom_data[index];
  }
}

template <typename Ftype, typename Btype>
void UpsampleLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top) {
  const Ftype* bottom_data = bottom[0]->gpu_data();
  const Ftype* bottom_mask = bottom[1]->gpu_data();
  Ftype* top_data = top[0]->mutable_gpu_data();
  caffe_gpu_set(top[0]->count(), Ftype(0), top_data);
  int bottom_count = bottom[0]->count();
  UpsampleForward<Ftype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
      bottom_count, bottom[0]->width(), bottom[0]->height(), 
      top[0]->width(), top[0]->height(), bottom_data, bottom_mask, top_data);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Ftype, typename Btype>
  __global__ void UpsampleBackward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Btype* top_diff,
      const Btype* bottom_mask, Btype* bottom_diff) {
    CUDA_KERNEL_LOOP(index, nthreads) {
      int offset = index / (in_w * in_h) * out_w * out_h;
      int upsample_idx = static_cast<int>(bottom_mask[index]);
      bottom_diff[index] = top_diff[offset + upsample_idx];
    }
  }

template <typename Ftype, typename Btype>
void UpsampleLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom) {
  if (propagate_down[0]) {
    const Btype* top_diff = top[0]->gpu_diff();
    const Btype* bottom_mask = bottom[1]->gpu_data();
    Btype* bottom_diff = bottom[0]->mutable_gpu_diff();
    const int bottom_count = bottom[0]->count();
    caffe_gpu_set(bottom_count, Btype(0.), bottom_diff);
    UpsampleBackward<Btype><<<CAFFE_GET_BLOCKS(bottom_count), CAFFE_CUDA_NUM_THREADS>>>(
        bottom_count, bottom[0]->width(), bottom[0]->height(), 
        top[0]->width(), top[0]->height(), top_diff, bottom_mask, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS_FB(UpsampleLayer);


}  // namespace caffe
