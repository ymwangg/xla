#include "curand_uniform.h"

#include "tensorflow/compiler/xla/service/custom_call_status.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla_custom_cuda_ops {

std::atomic<CurandGeneratorClient*> g_custom_uniform(nullptr);
std::once_flag g_custom_uniform_once;

CurandGeneratorClient::CurandGeneratorClient() {
  curandCreateGenerator(&this->generator, CURAND_RNG_PSEUDO_PHILOX4_32_10);
}

CurandGeneratorClient* CurandGeneratorClient::Get() {
  std::call_once(g_custom_uniform_once,
                 [&]() { g_custom_uniform = new CurandGeneratorClient(); });
  return g_custom_uniform.load();
}

curandGenerator_t& CurandGeneratorClient::GetCurandGenerator() {
  return this->generator;
}

void CurandGeneratorClient::SetRandomSeed(unsigned long long seed) {
  curandSetPseudoRandomGeneratorSeed(this->generator, seed);
}

absl::Status cuda_curand_uniform_f32_(CUstream stream, void** buffers,
                                      const char* opaque, size_t opaque_len) {
  float* output = reinterpret_cast<float*>(buffers[0]);

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
  CUDA_RETURN_IF_ERROR(
      CUDA_AS_STATUS(curandGenerateUniform(generator, output, num_elements)));
  return absl::OkStatus();
}

void cuda_curand_uniform_f32(CUstream stream, void** buffers,
                             const char* opaque, size_t opaque_len,
                             XlaCustomCallStatus* status) {
  auto result = cuda_curand_uniform_f32_(stream, buffers, opaque, opaque_len);
  if (!result.ok()) {
    absl::string_view message = result.message();
    XlaCustomCallStatusSetFailure(status, message.data(), message.length());
  }
}

XLA_REGISTER_CUSTOM_CALL_TARGET_WITH_SYM("cuda_curand_uniform_f32",
                                         cuda_curand_uniform_f32, "CUDA");

}  // namespace xla_custom_cuda_ops
