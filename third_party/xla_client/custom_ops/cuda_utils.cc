#include "cuda_utils.h"

namespace xla_custom_cuda_ops {

std::string ErrorString(curandStatus_t status) {
  switch (status) {
    case CURAND_STATUS_SUCCESS:
      return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
      return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
      return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
      return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
      return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
      return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
      return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
      return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
      return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
      return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
      return "CURAND_STATUS_INTERNAL_ERROR";
    default:
      return "UNKNOWN_ERROR";
  }
}

std::string ErrorString(cudaError_t error) { return cudaGetErrorString(error); }

template <typename T>
std::string ErrorString(T status, const char* file, std::int64_t line,
                        const char* expr) {
  return absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr,
                         ErrorString(status));
}

absl::Status AsStatus(cudaError_t error, const char* file, std::int64_t line,
                      const char* expr) {
  if (error != cudaSuccess)
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

absl::Status AsStatus(curandStatus_t error, const char* file, std::int64_t line,
                      const char* expr) {
  if (error != CURAND_STATUS_SUCCESS)
    return absl::InternalError(ErrorString(error, file, line, expr));
  return absl::OkStatus();
}

}  // namespace xla_custom_cuda_ops
