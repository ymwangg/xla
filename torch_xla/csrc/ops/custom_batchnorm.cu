#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <iostream>

#include "custom.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

__global__ struct sine : public thrust::unary_function<float, float> {
  float* dt_;
  sine(float* dt) : dt_(dt) {}
  __device__ float operator()(float x) { return *dt_; }
};

void xla_custom_batchnorm(CUstream stream, void** buffers, const char* opaque,
                          size_t opaque_len) {
  std::cout << "xla_custom_batchnorm" << std::endl;
  float* output = reinterpret_cast<float*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (size_t i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
    std::cout << shape.dimensions(i) << std::endl;
  }
  // std::call_once(cuda_rng, []() {
  //   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  //   curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  // });
  // curandSetStream(gen, stream);
  // curandGenerateUniform(gen, output, len);
  // std::cout << "done rng" << std::endl;
  // thrust::device_ptr<float> o = thrust::device_pointer_cast(output);
  // thrust::transform(thrust::cuda::par.on(stream), o, o + len, o,
  // bernoulli_fast(0.1));
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_batchnorm, "CUDA");

void xla_custom_batchnorm_backward(CUstream stream, void** buffers,
                                   const char* opaque, size_t opaque_len) {
  std::cout << "xla_custom_batchnorm_backward" << std::endl;
  float* output = reinterpret_cast<float*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (size_t i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
    std::cout << shape.dimensions(i) << std::endl;
  }
  // std::call_once(cuda_rng, []() {
  //   curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  //   curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  // });
  // curandSetStream(gen, stream);
  // curandGenerateUniform(gen, output, len);
  // std::cout << "done rng" << std::endl;
  // thrust::device_ptr<float> o = thrust::device_pointer_cast(output);
  // thrust::transform(thrust::cuda::par.on(stream), o, o + len, o,
  // bernoulli_fast(0.1));
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_batchnorm_backward, "CUDA");