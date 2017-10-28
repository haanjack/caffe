#ifndef CAFFE_UPSAMPLE_LAYER_HPP_
#define CAFFE_UPSAMPLE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

//#include "caffe/layers/upsample_layer.hpp"

namespace caffe {


template <typename Ftype, typename Btype>
class UpsampleLayer : public Layer<Ftype, Btype> {
 public:
  explicit UpsampleLayer(const LayerParameter& param)
      : Layer<Ftype, Btype>(param) {}
  virtual void LayerSetUp(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual inline const char* type() const { return "Upsample"; }
  virtual inline int ExactNumBottomBlobs() const { return 2; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,
      const vector<Blob*>& top);

  virtual void Backward_cpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);
  virtual void Backward_gpu(const vector<Blob*>& top,
      const vector<bool>& propagate_down, const vector<Blob*>& bottom);

/*
  virtual void UpsampleForward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* bottom_data,
      const Dtype* bottom_mask, Dtype* top_data);
  virtual void UpsampleBackward(const int nthreads, int in_w, int in_h,
      int out_w, int out_h, const Dtype* top_diff,
      const Dtype* bottom_mask, Dtype* bottom_diff);
*/

  int channels_;
  int height_;
  int width_;
  int scale_h_, scale_w_;
  bool pad_out_h_, pad_out_w_;
  int upsample_h_, upsample_w_;


};

}  // namespace caffe

#endif  
