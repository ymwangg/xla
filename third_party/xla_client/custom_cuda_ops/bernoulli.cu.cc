#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/bernoulli.h"

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>


namespace xla_custom_cuda_ops {

struct bernoulli {
  __host__ __device__ float operator()(const float& probability,
                                       const float& value) const {
    return value < probability ? 1.0 : 0.0;
  }
};

void bernoulli_compare(CUstream stream, const float* probability, float* output, int64_t num_elements) {
  thrust::device_ptr<const float> p = thrust::device_pointer_cast(probability);
  thrust::device_ptr<float> o = thrust::device_pointer_cast(output);
  thrust::transform(thrust::cuda::par.on(stream), p, p + num_elements, o, o,
                    bernoulli());
}

}  // namespace xla_custom_cuda_ops
