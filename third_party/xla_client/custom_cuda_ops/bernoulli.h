#ifndef XLA_CUSTOM_BERNOULLI_
#define XLA_CUSTOM_BERNOULLI_

#include <cuda.h>

namespace xla_custom_cuda_ops {

void bernoulli_compare(CUstream stream, const float* probability, float* output, int64_t num_elements);

}  // namespace xla_custom_cuda_ops

#endif  // XLA_CUSTOM_BERNOULLI_
