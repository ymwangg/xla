#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/curand_uniform.h"

#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/cuda_utils.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla_custom_cuda_ops {

CurandGeneratorClient::CurandGeneratorClient() {
  curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}

CurandGeneratorClient* CurandGeneratorClient::Get() {
  static auto* instance = new CurandGeneratorClient;
  return instance;
}

curandGenerator_t& CurandGeneratorClient::GetCurandGenerator() {
  return this->generator;
}

void CurandGeneratorClient::SetRngSeed(unsigned long long seed) {
  curandSetPseudoRandomGeneratorSeed(this->generator, seed);
}

absl::Status curand_uniform_(CUstream stream, void** buffers,
                             const char* opaque, size_t opaque_len) {
  void* output = reinterpret_cast<void*>(buffers[0]);

  xla::ShapeProto shape;
  if (!shape.ParseFromArray(opaque, opaque_len)) {
    return absl::InternalError("Failed parsing the shape proto");
  }
  int64_t num_elements = 1;
  for (auto dim : shape.dimensions()) {
    num_elements *= dim;
  }

  curandGenerator_t& generator =
      CurandGeneratorClient::Get()->GetCurandGenerator();
  CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandSetStream(generator, stream)));
  switch (shape.element_type()) {
    case xla::PrimitiveType::F32:
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniform(
          generator, reinterpret_cast<float*>(output), num_elements)));
      break;
    case xla::PrimitiveType::F64:
      CUDA_RETURN_IF_ERROR(CUDA_AS_STATUS(curandGenerateUniformDouble(
          generator, reinterpret_cast<double*>(output), num_elements)));
      break;
    default:
      std::string type_name =
          xla::primitive_util::LowercasePrimitiveTypeName(shape.element_type());
      return absl::InternalError(
          absl::StrCat("Unsupported data type: ", type_name));
  }
  return absl::OkStatus();
}

void curand_uniform(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len, XlaCustomCallStatus* status) {
  auto result = curand_uniform_(stream, buffers, opaque, opaque_len);
  if (!result.ok()) {
    absl::string_view message = result.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("custom_curand_uniform",
                                         curand_uniform, "CUDA");

}  // namespace xla_custom_cuda_ops
