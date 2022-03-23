#ifndef XLA_CUSTOM_CUDA_UTILS_
#define XLA_CUSTOM_CUDA_UTILS_

#include "absl/base/casts.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "third_party/gpus/cuda/include/curand.h"

#define CUDA_AS_STATUS(expr) \
  xla_custom_cuda_ops::AsStatus(expr, __FILE__, __LINE__, #expr)

#define CUDA_RETURN_IF_ERROR(expr) \
  {                                \
    auto s___ = (expr);            \
    if (!s___.ok()) return s___;   \
  }

namespace xla_custom_cuda_ops {

// Used via CUDA_AS_STATUS(expr) macro.
absl::Status AsStatus(cudaError_t error, const char* file, std::int64_t line,
                      const char* expr);

absl::Status AsStatus(curandStatus_t error, const char* file, std::int64_t line,
                      const char* expr);

}  // namespace xla_custom_cuda_ops

#endif  // XLA_CUSTOM_CUDA_UTILS_
