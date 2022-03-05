#include <cuda.h>
#include <cuda_fp16.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/transform.h>

#include <iostream>

#include "custom.h"
#include "tensorflow/compiler/xla/service/custom_call_target_registry.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

__global__ void saxpy(int n, float a, float* x, float* y) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}

void test() {
  int N = 1 << 20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N * sizeof(float));
  y = (float*)malloc(N * sizeof(float));

  cudaMalloc(&d_x, N * sizeof(float));
  cudaMalloc(&d_y, N * sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

  // Perform SAXPY on 1M elements
  saxpy<<<(N + 255) / 256, 256>>>(N, 2.0f, d_x, d_y);

  cudaMemcpy(y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++) maxError = max(maxError, abs(y[i] - 4.0f));
  printf("Max error: %f\n", maxError);

  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}

__global__ void custom_call_kernel(const float* input, const int64_t* index,
                                   float* out, const int64_t len,
                                   const int64_t dim) {
  const int n = 10;
  for (size_t thread_id = n * threadIdx.x; thread_id + n < len;
       thread_id += n * blockDim.x) {
    for (int k = 0; k < n; k++) {
      int64_t idx = index[thread_id + k];
      for (size_t i = 0; i < dim; i++) {
        out[(thread_id + k) * dim + i] = input[idx * dim + i];
      }
    }
  }
}

void do_custom_call(CUstream stream, void** buffers, const char* opaque,
                    size_t opaque_len) {
  auto start = std::chrono::steady_clock::now();
  const float* input = reinterpret_cast<const float*>(buffers[0]);
  const int64_t* index = reinterpret_cast<const int64_t*>(buffers[1]);
  float* output = reinterpret_cast<float*>(buffers[2]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  const int64_t len = shape.dimensions(0);
  const int64_t dim = shape.dimensions(1);
  const int64_t block_dim = 1024;
  const int64_t grid_dim = 1;
  // custom_call_kernel<<<grid_dim, block_dim,
  //                      /*dynamic_shared_mem_bytes=*/0, stream>>>(input,
  //                      index, output, len, dim);
  auto end = std::chrono::steady_clock::now();
  std::cout << "dt="
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_call, "CUDA");

static curandGenerator_t gen;
static std::once_flag cuda_rng;

void xla_custom_rand_float(CUstream stream, void** buffers, const char* opaque,
                           size_t opaque_len) {
  // auto start = std::chrono::steady_clock::now();
  float* output = reinterpret_cast<float*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (int i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
  }
  // std::cout << "len=" << len << std::endl;
  std::call_once(cuda_rng, []() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  });
  curandSetStream(gen, stream);
  curandGenerateUniform(gen, output, len);
  // auto end = std::chrono::steady_clock::now();
  // std::cout << "dt="
  //           << std::chrono::duration_cast<std::chrono::microseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << " us" << std::endl;
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_rand_float, "CUDA");

__global__ void set_value(float* output, const int64_t len) {
  for (size_t thread_id = threadIdx.x; thread_id < len;
       thread_id += blockDim.x) {
    output[thread_id] = 0.5;
  }
}

void do_custom_rand2(CUstream stream, void** buffers, const char* opaque,
                     size_t opaque_len) {
  auto start = std::chrono::steady_clock::now();
  float* output = reinterpret_cast<float*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (int i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
  }
  std::cout << "len=" << len << std::endl;
  // const int64_t block_dim = 1024;
  // const int64_t grid_dim = 1;
  // set_value<<<grid_dim, block_dim,
  //             /*dynamic_shared_mem_bytes=*/0, stream>>>(output, len);
  std::call_once(cuda_rng, []() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  });
  curandGenerateUniform(gen, output, len);
  auto end = std::chrono::steady_clock::now();
  std::cout << "dt="
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;
  float* tmp = (float*)malloc(len * sizeof(float));
  cudaMemcpy(tmp, output, len * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < len; i++) {
    printf("%1.4f ", tmp[i]);
  }
  printf("\n");
}
XLA_REGISTER_CUSTOM_CALL_TARGET(do_custom_rand2, "CUDA");

static curandState* states;

__global__ void setup_kernel(curandState* states) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  curand_init(1234, idx, 0, &states[idx]);
}

__global__ void generate_uniform(__half* output, curandState* states,
                                 int64_t len, int64_t num_per_thread) {
  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = 0; i < num_per_thread; i++) {
    int64_t out_idx = idx * num_per_thread + i;
    if (out_idx < len) {
      output[out_idx] = __float2half(curand_uniform(&states[idx]));
    }
  }
}

void xla_custom_rand_half(CUstream stream, void** buffers, const char* opaque,
                          size_t opaque_len) {
  auto start = std::chrono::steady_clock::now();
  __half* output = reinterpret_cast<__half*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (size_t i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
  }
  std::cout << "len=" << len << std::endl;
  int64_t N = 512;
  std::call_once(cuda_rng, [&]() {
    cudaMalloc((void**)&states, N * sizeof(curandState_t));
    setup_kernel<<<1, N, 0, stream>>>(states);
  });
  int64_t num_per_thread = (len + N - 1) / N;
  generate_uniform<<<1, N, 0, stream>>>(output, states, len, num_per_thread);
  auto end = std::chrono::steady_clock::now();
  std::cout << "dt="
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << " us" << std::endl;
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_rand_half, "CUDA");

struct bernoulli {
  __host__ __device__ float operator()(const float& probability,
                                       const float& value) const {
    return value < probability ? 1.0f : 0.0f;
  }
};

void xla_custom_bernoulli(CUstream stream, void** buffers, const char* opaque,
                          size_t opaque_len) {
  const float* probability = reinterpret_cast<const float*>(buffers[0]);
  float* output = reinterpret_cast<float*>(buffers[1]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (size_t i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
  }
  std::call_once(cuda_rng, []() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  });
  curandSetStream(gen, stream);
  curandGenerateUniform(gen, output, len);
  thrust::device_ptr<const float> p = thrust::device_pointer_cast(probability);
  thrust::device_ptr<float> o = thrust::device_pointer_cast(output);
  thrust::transform(thrust::cuda::par.on(stream), p, p + len, o, o,
                    bernoulli());
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_bernoulli, "CUDA");

struct bernoulli_fast {
  const float probability_;
  bernoulli_fast(float probability) : probability_(probability) {}
  __host__ __device__ float operator()(float& value) const {
    return value < probability_ ? 0.0f : 1.0f;
  }
};

void xla_custom_bernoulli_fast(CUstream stream, void** buffers,
                               const char* opaque, size_t opaque_len) {
  std::cout << "fast bernoulli" << std::endl;
  float* output = reinterpret_cast<float*>(buffers[0]);
  xla::ShapeProto shape;
  shape.ParseFromArray(opaque, opaque_len);
  int64_t len = 1;
  for (size_t i = 0; i < shape.dimensions().size(); i++) {
    len *= shape.dimensions(i);
  }
  std::cout << "len=" << len << std::endl;
  std::call_once(cuda_rng, []() {
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  });
  curandSetStream(gen, stream);
  curandGenerateUniform(gen, output, len);
  std::cout << "done rng" << std::endl;
  thrust::device_ptr<float> o = thrust::device_pointer_cast(output);
  thrust::transform(thrust::cuda::par.on(stream), o, o + len, o, bernoulli_fast(0.1));
}
XLA_REGISTER_CUSTOM_CALL_TARGET(xla_custom_bernoulli_fast, "CUDA");
