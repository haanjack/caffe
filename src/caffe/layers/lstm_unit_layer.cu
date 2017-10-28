#include <algorithm>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/lstm_layer.hpp"

namespace caffe {

template <typename Ftype, typename Btype>
__device__ Ftype sigmoid(const Ftype x) {
  return Ftype(1) / (Ftype(1) + exp(-x));
}

template <typename Ftype, typename Btype>
__device__ Ftype tanh(const Ftype x) {
  return Ftype(2) * sigmoid(Ftype(2) * x) - Ftype(1);
}

template <typename Ftype, typename Btype>
__global__ void LSTMActsForward(const int nthreads, const int dim,
                                const Ftype* X, Ftype* X_acts) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int x_dim = 4 * dim;
    const int d = index % x_dim;
    if (d < 3 * dim) {
      X_acts[index] = sigmoid(X[index]);
    } else {
      X_acts[index] = tanh(X[index]);
    }
  }
}

template <typename Ftype, typename Btype>
__global__ void LSTMUnitForward(const int nthreads, const int dim,
    const Ftype* C_prev, const Ftype* X, const Ftype* cont,
    Ftype* C, Ftype* H) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Ftype* X_offset = X + 4 * dim * n;
    const Ftype i = X_offset[d];
    const Ftype f = X_offset[1 * dim + d];
    const Ftype o = X_offset[2 * dim + d];
    const Ftype g = X_offset[3 * dim + d];
    const Ftype c_prev = C_prev[index];
    const Ftype c = cont[n] * f * c_prev + i * g;
    C[index] = c;
    const Ftype tanh_c = tanh(c);
    H[index] = o * tanh_c;
  }
}

template <typename Ftype, typename Btype>
void LSTMUnitLayer<Ftype, Btype>::Forward_gpu(const vector<Blob*>& bottom,
    const vector<Blob*>& top) {
  const int count = top[1]->count();
  const Ftype* C_prev = bottom[0]->gpu_data();
  const Ftype* X = bottom[1]->gpu_data();
  const Ftype* cont = bottom[2]->gpu_data();
  Ftype* X_acts = X_acts_.mutable_gpu_data();
  Ftype* C = top[0]->mutable_gpu_data();
  Ftype* H = top[1]->mutable_gpu_data();
  const int X_count = bottom[1]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  LSTMActsForward<Ftype><<<CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS>>>(
      X_count, hidden_dim_, X, X_acts);
  CUDA_POST_KERNEL_CHECK;
  // NOLINT_NEXT_LINE(whitespace/operators)
  LSTMUnitForward<Ftype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, hidden_dim_, C_prev, X_acts, cont, C, H);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Ftype, typename Btype>
__global__ void LSTMUnitBackward(const int nthreads, const int dim,
    const Btype* C_prev, const Btype* X, const Btype* C, const Btype* H,
    const Btype* cont, const Btype* C_diff, const Btype* H_diff,
    Btype* C_prev_diff, Btype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int n = index / dim;
    const int d = index % dim;
    const Btype* X_offset = X + 4 * dim * n;
    const Btype i = X_offset[d];
    const Btype f = X_offset[1 * dim + d];
    const Btype o = X_offset[2 * dim + d];
    const Btype g = X_offset[3 * dim + d];
    const Btype c_prev = C_prev[index];
    const Btype c = C[index];
    const Btype tanh_c = tanh(c);
    Btype* c_prev_diff = C_prev_diff + index;
    Btype* X_diff_offset = X_diff + 4 * dim * n;
    Btype* i_diff = X_diff_offset + d;
    Btype* f_diff = X_diff_offset + 1 * dim + d;
    Btype* o_diff = X_diff_offset + 2 * dim + d;
    Btype* g_diff = X_diff_offset + 3 * dim + d;
    const Btype c_term_diff =
        C_diff[index] + H_diff[index] * o * (1 - tanh_c * tanh_c);
    const Btype cont_n = cont[n];
    *c_prev_diff = cont_n * c_term_diff * f;
    *i_diff = c_term_diff * g;
    *f_diff = cont_n * c_term_diff * c_prev;
    *o_diff = H_diff[index] * tanh_c;
    *g_diff = c_term_diff * i;
  }
}

template <typename Ftype, typename Btype>
__global__ void LSTMActsBackward(const int nthreads, const int dim,
    const Btype* X_acts, const Btype* X_acts_diff, Btype* X_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    const int x_dim = 4 * dim;
    const int d = index % x_dim;
    const Btype X_act = X_acts[index];
    if (d < 3 * dim) {
      X_diff[index] = X_acts_diff[index] * X_act * (Btype(1) - X_act);
    } else {
      X_diff[index] = X_acts_diff[index] * (Btype(1) - X_act * X_act);
    }
  }
}

template <typename Ftype, typename Btype>
void LSTMUnitLayer<Ftype, Btype>::Backward_gpu(const vector<Blob*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob*>& bottom) {
  CHECK(!propagate_down[2]) << "Cannot backpropagate to sequence indicators.";
  if (!propagate_down[0] && !propagate_down[1]) { return; }

  const int count = top[1]->count();
  const Btype* C_prev = bottom[0]->gpu_data();
  const Btype* X_acts = X_acts_.gpu_data();
  const Btype* cont = bottom[2]->gpu_data();
  const Btype* C = top[0]->gpu_data();
  const Btype* H = top[1]->gpu_data();
  const Btype* C_diff = top[0]->gpu_diff();
  const Btype* H_diff = top[1]->gpu_diff();
  Btype* C_prev_diff = bottom[0]->mutable_gpu_diff();
  Btype* X_acts_diff = X_acts_.mutable_gpu_diff();
  LSTMUnitBackward<Btype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(count, hidden_dim_,
      C_prev, X_acts, C, H, cont, C_diff, H_diff, C_prev_diff, X_acts_diff);
  CUDA_POST_KERNEL_CHECK;
  const int X_count = bottom[1]->count();
  Btype* X_diff = bottom[1]->mutable_gpu_diff();
  LSTMActsBackward<Btype>  // NOLINT_NEXT_LINE(whitespace/operators)
      <<<CAFFE_GET_BLOCKS(X_count), CAFFE_CUDA_NUM_THREADS>>>(
      X_count, hidden_dim_, X_acts, X_acts_diff, X_diff);
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(LSTMUnitLayer);

}  // namespace caffe
