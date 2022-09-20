#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/bernoulli.h"

#include "rectangular_lsap.h"
#include "lsainfo.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/cuda_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla_custom_cuda_ops {

CurandContext::CurandContext() {
  curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
  this->seed_ = 1234ULL;
  this->curr_offset_ = 0ULL;
}

CurandContext* CurandContext::Get() {
  static auto* instance = new CurandContext;
  return instance;
}

curandGenerator_t& CurandContext::GetCurandGenerator() {
  return this->generator;
}

void CurandContext::SetRngSeed(uint64_t seed) {
  curandSetPseudoRandomGeneratorSeed(this->generator, seed);
  this->seed_ = seed;
}

std::pair<uint64_t, uint64_t> CurandContext::IncrementOffset(
    uint64_t num_elements, uint64_t num_threads) {
  uint64_t offset_inc = (num_elements + num_threads - 1) / num_threads;
  uint64_t offset = this->curr_offset_;
  this->curr_offset_ += offset_inc;
  return std::pair<uint64_t, uint64_t>(this->seed_, offset);
}

absl::Status XlaCustomBernoulliCuda_(CUstream stream, void** buffers,
                                     const char* opaque, size_t opaque_len) {
  const void* probability = reinterpret_cast<const void*>(buffers[0]);
  void* output = reinterpret_cast<void*>(buffers[1]);

  xla::ShapeProto shape;
  if (!shape.ParseFromArray(opaque, opaque_len)) {
    return absl::InternalError("Failed parsing the shape proto");
  }
  int64_t num_elements = 1;
  for (auto dim : shape.dimensions()) {
    num_elements *= dim;
  }

  curandGenerator_t& generator = CurandContext::Get()->GetCurandGenerator();
  CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandSetStream(generator, stream)));

  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
      // std::cout << "F16" << std::endl;
      LaunchBernoulliKernelHalf(
          stream, reinterpret_cast<const __half*>(probability),
          reinterpret_cast<__half*>(output), num_elements);
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F32:
      // std::cout << "F32" << std::endl;
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniform(
          generator, reinterpret_cast<float*>(output), num_elements)));
      LaunchBernoulliKernel<float>(
          stream, reinterpret_cast<const float*>(probability),
          reinterpret_cast<float*>(output), num_elements);
      // LaunchBernoulliKernel2(stream,
      //                        reinterpret_cast<const float*>(probability),
      //                        reinterpret_cast<float*>(output), num_elements);
      // CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F64:
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniformDouble(
          generator, reinterpret_cast<double*>(output), num_elements)));
      LaunchBernoulliKernel<double>(
          stream, reinterpret_cast<const double*>(probability),
          reinterpret_cast<double*>(output), num_elements);
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    default:
      std::string type_name =
          xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type());
      return absl::InternalError(
          absl::StrCat("Unsupported data type: ", type_name));
  }
  return absl::OkStatus();
}

void XlaCustomBernoulliCuda(CUstream stream, void** buffers, const char* opaque,
                            size_t opaque_len, XlaCustomCallStatus* status) {
  auto result = XlaCustomBernoulliCuda_(stream, buffers, opaque, opaque_len);
  if (!result.ok()) {
    absl::string_view message = result.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("XlaCustomBernoulliCuda",
                                         XlaCustomBernoulliCuda, "CUDA");

absl::Status XlaCustomRngCuda_(CUstream stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  void* output = reinterpret_cast<void*>(buffers[1]);

  xla::ShapeProto shape;
  if (!shape.ParseFromArray(opaque, opaque_len)) {
    return absl::InternalError("Failed parsing the shape proto");
  }
  int64_t num_elements = 1;
  for (auto dim : shape.dimensions()) {
    num_elements *= dim;
  }

  curandGenerator_t& generator = CurandContext::Get()->GetCurandGenerator();
  CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandSetStream(generator, stream)));

  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
      // std::cout << "F16" << std::endl;
      LaunchRngHalf(stream, reinterpret_cast<__half*>(output), num_elements);
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F32:
      // std::cout << "F32" << std::endl;
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniform(
          generator, reinterpret_cast<float*>(output), num_elements)));
      // LaunchRng(stream, reinterpret_cast<float*>(output), num_elements);
      // CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F64:
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniformDouble(
          generator, reinterpret_cast<double*>(output), num_elements)));
      // LaunchBernoulliKernel<double>(
      //     stream, reinterpret_cast<const double*>(probability),
      //     reinterpret_cast<double*>(output), num_elements);
      // CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    default:
      std::string type_name =
          xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type());
      return absl::InternalError(
          absl::StrCat("Unsupported data type: ", type_name));
  }
  return absl::OkStatus();
}

void XlaCustomRngCuda(CUstream stream, void** buffers, const char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  auto result = XlaCustomRngCuda_(stream, buffers, opaque, opaque_len);
  if (!result.ok()) {
    absl::string_view message = result.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("XlaCustomRngCuda", XlaCustomRngCuda,
                                         "CUDA");

absl::Status XlaCustomDropoutCuda_(CUstream stream, void** buffers,
                                   const char* opaque, size_t opaque_len) {
  const void* input = reinterpret_cast<const void*>(buffers[0]);
  void* output = reinterpret_cast<void*>(buffers[1]);
  uint8_t* mask = reinterpret_cast<uint8_t*>(buffers[2]);
  // std::cout << input << "," << (void*)mask << "," << output << std::endl;

  xla::ShapeProto shape;
  if (!shape.ParseFromArray(opaque, opaque_len)) {
    return absl::InternalError("Failed parsing the shape proto");
  }
  int64_t num_elements = 1;
  for (auto dim : shape.dimensions()) {
    num_elements *= dim;
  }

  curandGenerator_t& generator = CurandContext::Get()->GetCurandGenerator();
  CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandSetStream(generator, stream)));

  switch (shape.element_type()) {
    case xla::PrimitiveType::F16:
      // std::cout << "F16," << num_elements << std::endl;
      LaunchDropoutKernelHalf(stream, reinterpret_cast<const __half*>(input),
                              reinterpret_cast<__half*>(output), mask,
                              num_elements);
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F32:
      // std::cout << "F32," << num_elements << std::endl;
      LaunchDropoutKernel(stream, reinterpret_cast<const float*>(input),
                          reinterpret_cast<float*>(output), mask, num_elements);
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    case xla::PrimitiveType::F64:
      // CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniformDouble(
      //     generator, reinterpret_cast<double*>(output), num_elements)));
      // LaunchBernoulliKernel<double>(
      //     stream, reinterpret_cast<const double*>(probability),
      //     reinterpret_cast<double*>(output), num_elements);
      // CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(cudaGetLastError()));
      break;
    default:
      std::string type_name =
          xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type());
      return absl::InternalError(
          absl::StrCat("Unsupported data type: ", type_name));
  }
  return absl::OkStatus();
}

void XlaCustomDropoutCuda(CUstream stream, void** buffers, const char* opaque,
                          size_t opaque_len, XlaCustomCallStatus* status) {
  auto result = XlaCustomDropoutCuda_(stream, buffers, opaque, opaque_len);
  if (!result.ok()) {
    absl::string_view message = result.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}
XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("XlaCustomDropoutCuda",
                                         XlaCustomDropoutCuda, "CUDA");

void LinearSumAssignment(CUstream stream, void** buffers, const char* opaque,
                         size_t opaque_len, XlaCustomCallStatus* status) {
  const void* input = reinterpret_cast<const void*>(buffers[0]);
  int64_t* row_idx = reinterpret_cast<int64_t*>(buffers[1]);
  int64_t* col_idx = reinterpret_cast<int64_t*>(buffers[2]);
  // std::cout << input << "," << (void*)mask << "," << output << std::endl;

  if (opaque_len != sizeof(xla::LSAInfo)) {
    return;
  }
  xla::LSAInfo* LSA = absl::bit_cast<xla::LSAInfo*>(opaque);
  xla::Shape shape = LSA->input_shape;
  bool maximize = LSA->maximize;
  
  int64_t min_dim = shape.dimensions(0) < shape.dimensions(1)
                        ? shape.dimensions(0)
                        : shape.dimensions(1);
  int64_t num_elements = 1;
  for (auto dim : shape.dimensions()) {
    num_elements *= dim;
  }
  std::vector<int64_t> row_idx_h;
  std::vector<int64_t> col_idx_h;
  std::vector<float> cost_h;
  row_idx_h.resize(min_dim);
  col_idx_h.resize(min_dim);
  cost_h.resize(num_elements);
  // std::cout << "num_elements = " << num_elements << " min_dim=" << min_dim
  //           << std::endl;
  cudaMemcpy(cost_h.data(), input, num_elements * sizeof(float),
             cudaMemcpyDeviceToHost);
  // for (int i = 0; i < num_elements; i++) {
  //   if (i != 0 && i % shape.dimensions(1) == 0) {
  //     std::cout << std::endl;
  //   }
  //   std::cout << cost_h[i] << " ";
  // }
  // std::cout << std::endl;
  solve_rectangular_linear_sum_assignment(
      shape.dimensions(0), shape.dimensions(1), cost_h.data(), maximize,
      row_idx_h.data(), col_idx_h.data());
  cudaMemcpy(row_idx, row_idx_h.data(), min_dim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_idx, col_idx_h.data(), min_dim * sizeof(int64_t),
             cudaMemcpyHostToDevice);
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("LinearSumAssignment",
                                         LinearSumAssignment, "CUDA");
}  // namespace xla_custom_cuda_ops
