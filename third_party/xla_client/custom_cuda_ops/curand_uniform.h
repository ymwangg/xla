#ifndef XLA_CUSTOM_CURAND_UNIFORM_
#define XLA_CUSTOM_CURAND_UNIFORM_

#include <curand.h>

namespace xla_custom_cuda_ops {

// A helper class to store curandGenerator
class CurandGeneratorClient {
 public:
  static CurandGeneratorClient* Get();
  curandGenerator_t& GetCurandGenerator();
  void SetRngSeed(unsigned long long seed);

 private:
  curandGenerator_t generator;
  CurandGeneratorClient();
  ~CurandGeneratorClient() = default;
  CurandGeneratorClient(const CurandGeneratorClient&) = delete;
  CurandGeneratorClient& operator=(const CurandGeneratorClient&) = delete;
};

}  // namespace xla_custom_cuda_ops

#endif  // XLA_CUSTOM_CURAND_UNIFORM_
