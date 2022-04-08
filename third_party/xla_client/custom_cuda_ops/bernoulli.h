#ifndef XLA_CUSTOM_BERNOULLI_
#define XLA_CUSTOM_BERNOULLI_

#include <cuda.h>
#include <curand.h>
#include <cuda_fp16.h>

#include <algorithm>

#define XLA_CUDA_NUM_THREADS 512
#define XLA_MAXIMUM_NUM_BLOCKS 262144

namespace xla_custom_cuda_ops {

inline int GET_BLOCKS(const int N) {
  return (std::max)(
      (std::min)((N + XLA_CUDA_NUM_THREADS - 1) / XLA_CUDA_NUM_THREADS,
                 XLA_MAXIMUM_NUM_BLOCKS),
      // Use at least 1 block, since CUDA does not allow empty block
      1);
}

template <typename T>
void LaunchBernoulliKernel(CUstream stream, const T* probability, T* value,
                           int64_t num_elements);
void LaunchBernoulliKernel2(CUstream stream, const float* probability,
                            float* value, uint64_t num_elements);
void LaunchBernoulliKernelHalf(CUstream stream, const __half* probability,
                               __half* value, uint64_t num_elements);
void LaunchRng(CUstream stream, float* output, uint64_t num_elements);
void LaunchRngHalf(CUstream stream, __half* output, uint64_t num_elements);

// A helper class to store curandGenerator
class CurandContext {
 public:
  static CurandContext* Get();
  curandGenerator_t& GetCurandGenerator();
  void SetRngSeed(uint64_t seed);
  std::pair<uint64_t, uint64_t> IncrementOffset(uint64_t num_elements,
                                                uint64_t num_threads);

 private:
  curandGenerator_t generator;
  CurandContext();
  ~CurandContext() = default;
  CurandContext(const CurandContext&) = delete;
  CurandContext& operator=(const CurandContext&) = delete;
  uint64_t seed_;
  uint64_t curr_offset_;
};

}  // namespace xla_custom_cuda_ops

#endif  // XLA_CUSTOM_BERNOULLI_
