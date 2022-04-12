#include "tensorflow/compiler/xla/xla_client/custom_cuda_ops/bernoulli.h"

#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace xla_custom_cuda_ops {

template <typename T>
struct bernoulli_functor {
  __host__ __device__ T operator()(const T& probability, const T& value) const {
    return value <= probability ? 1.0 : 0.0;
  }
};

template <typename T>
void LaunchBernoulliKernel(CUstream stream, const T* probability, T* value,
                           int64_t num_elements) {
  thrust::device_ptr<const T> p_ptr = thrust::device_pointer_cast(probability);
  thrust::device_ptr<T> v_ptr = thrust::device_pointer_cast(value);
  thrust::transform(thrust::cuda::par.on(stream), p_ptr, p_ptr + num_elements,
                    v_ptr, v_ptr, bernoulli_functor<T>());
}

__global__ void device_kernel(const float* probability, float* value,
                              int64_t num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < num_elements) {
    value[idx] = value[idx] <= probability[idx] ? 1.0 : 0.0;
  }
}

template void LaunchBernoulliKernel(CUstream stream, const float* probability,
                                    float* value, int64_t num_elements);

template void LaunchBernoulliKernel(CUstream stream, const double* probability,
                                    double* value, int64_t num_elements);

#define CUDA_1D_KERNEL_LOOP(i, n)                                 \
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)

const int unroll_factor = 4;

__global__ void dropout_kernel(const float* probability, float* out,
                               std::pair<uint64_t, uint64_t> seed,
                               const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    float4 rand = curand_uniform4(&state);

    int i = j * unroll_factor;

    out[i] = rand.x < probability[i];
    out[i + 1] = rand.y < probability[i + 1];
    out[i + 2] = rand.z < probability[i + 2];
    out[i + 3] = rand.w < probability[i + 3];
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      out[i] = rand_data[k++] < probability[i];
    }
  }
}

__global__ void dropout_kernel_half(const __half* probability, __half* out,
                                    std::pair<uint64_t, uint64_t> seed,
                                    const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    float4 rand = curand_uniform4(&state);

    int i = j * unroll_factor;

    out[i] = __float2half(rand.x) < probability[i];
    out[i + 1] = __float2half(rand.y) < probability[i + 1];
    out[i + 2] = __float2half(rand.z) < probability[i + 2];
    out[i + 3] = __float2half(rand.w) < probability[i + 3];
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      out[i] = __float2half(rand_data[k++]) < probability[i];
    }
  }
}

void LaunchBernoulliKernel2(CUstream stream, const float* probability,
                            float* value, uint64_t num_elements) {
  // std::cout << "LaunchBernoulliKernel2" << std::endl;
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  // std::cout << "block_dim=" << block_dim << ",grid_dim=" << grid_dim
  //           << std::endl;
  // std::cout << "seed=(" << seed.first << "," << seed.second << ")" <<
  // std::endl;
  dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(probability, value, seed,
                                                     num_elements);
}

void LaunchBernoulliKernelHalf(CUstream stream, const __half* probability,
                               __half* value, uint64_t num_elements) {
  // std::cout << "LaunchBernoulliKernel2" << std::endl;
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  // std::cout << "block_dim=" << block_dim << ",grid_dim=" << grid_dim
  //           << std::endl;
  // std::cout << "seed=(" << seed.first << "," << seed.second << ")" <<
  // std::endl;
  dropout_kernel_half<<<grid_dim, block_dim, 0, stream>>>(probability, value,
                                                          seed, num_elements);
}

__global__ void rng_kernel_half(__half* out, std::pair<uint64_t, uint64_t> seed,
                                const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    float4 rand = curand_uniform4(&state);

    int i = j * unroll_factor;

    out[i] = __float2half(rand.x);
    out[i + 1] = __float2half(rand.y);
    out[i + 2] = __float2half(rand.z);
    out[i + 3] = __float2half(rand.w);
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      out[i] = __float2half(rand_data[k++]);
    }
  }
}

void LaunchRngHalf(CUstream stream, __half* output, uint64_t num_elements) {
  // std::cout << "LaunchBernoulliKernel2" << std::endl;
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  // std::cout << "block_dim=" << block_dim << ",grid_dim=" << grid_dim
  //           << std::endl;
  // std::cout << "seed=(" << seed.first << "," << seed.second << ")" <<
  // std::endl;
  rng_kernel_half<<<grid_dim, block_dim, 0, stream>>>(output, seed,
                                                      num_elements);
}

__global__ void rng_kernel(float* out, std::pair<uint64_t, uint64_t> seed,
                           const int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    float4 rand = curand_uniform4(&state);

    int i = j * unroll_factor;

    out[i] = rand.x;
    out[i + 1] = rand.y;
    out[i + 2] = rand.z;
    out[i + 3] = rand.w;
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      out[i] = rand_data[k++];
    }
  }
}

void LaunchRng(CUstream stream, float* output, uint64_t num_elements) {
  // std::cout << "LaunchBernoulliKernel2" << std::endl;
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  // std::cout << "block_dim=" << block_dim << ",grid_dim=" << grid_dim
  //           << std::endl;
  // std::cout << "seed=(" << seed.first << "," << seed.second << ")" <<
  // std::endl;
  rng_kernel<<<grid_dim, block_dim, 0, stream>>>(output, seed, num_elements);
}

__global__ void dropout_kernel_half(const __half* input, __half* out,
                                    uint8_t* mask,
                                    std::pair<uint64_t, uint64_t> seed,
                                    const int N) {
  const float ratio = 0.1;
  const float scale = 1. / (1. - ratio);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    int i = j * unroll_factor;

    const __half2* vals_half = reinterpret_cast<const __half2*>(input + i);
    float2 vals_half_f[2];
    vals_half_f[0] = __half22float2(vals_half[0]);
    vals_half_f[1] = __half22float2(vals_half[1]);

    uint8_t m[unroll_factor];
    float4 rand = curand_uniform4(&state);
    m[0] = (uint8_t)(rand.x > ratio);
    m[1] = (uint8_t)(rand.y > ratio);
    m[2] = (uint8_t)(rand.z > ratio);
    m[3] = (uint8_t)(rand.w > ratio);

    out[i] = __float2half(vals_half_f[0].x * scale * m[0]);
    out[i + 1] = __float2half(vals_half_f[0].y * scale * m[1]);
    out[i + 2] = __float2half(vals_half_f[1].x * scale * m[2]);
    out[i + 3] = __float2half(vals_half_f[1].y * scale * m[3]);

    mask[i] = m[0];
    mask[i + 1] = m[1];
    mask[i + 2] = m[2];
    mask[i + 3] = m[3];
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      uint8_t m = (uint8_t)(rand_data[k++] > ratio);
      out[i] = __float2half((float)input[i] * scale * m);
      mask[i] = m;
    }
  }
}

__global__ void dropout_kernel(const float* input, float* out, uint8_t* mask,
                               std::pair<uint64_t, uint64_t> seed,
                               const int N) {
  const float ratio = 0.1;
  const float scale = 1. / (1. - ratio);
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  curandStatePhilox4_32_10_t state;
  curand_init(seed.first, idx, seed.second, &state);
  uint8_t m[unroll_factor];

  CUDA_1D_KERNEL_LOOP(j, N / unroll_factor) {
    int i = j * unroll_factor;
    const float4* in_vals = reinterpret_cast<const float4*>(input + i);
    float4 rand = curand_uniform4(&state);

    m[0] = (uint8_t)(rand.x > ratio);
    m[1] = (uint8_t)(rand.y > ratio);
    m[2] = (uint8_t)(rand.z > ratio);
    m[3] = (uint8_t)(rand.w > ratio);

    out[i] = in_vals->x * scale * m[0];
    out[i + 1] = in_vals->y * scale * m[1];
    out[i + 2] = in_vals->z * scale * m[2];
    out[i + 3] = in_vals->w * scale * m[3];

    mask[i] = m[0];
    mask[i + 1] = m[1];
    mask[i + 2] = m[2];
    mask[i + 3] = m[3];
  }
  int high_index = ((((N / unroll_factor) - 1) / blockDim.x + 1) *
                    (unroll_factor * blockDim.x)) +
                   threadIdx.x;
  if (N > high_index) {
    float4 rand = curand_uniform4(&state);
    float* rand_data = &(rand.x);
    int k = 0;
    for (int i = high_index; i < N; i++) {
      uint8_t m = (uint8_t)(rand_data[k++] > ratio);
      out[i] = input[i] * scale * m;
      mask[i] = m;
    }
  }
}

void LaunchDropoutKernelHalf(CUstream stream, const __half* input,
                             __half* output, uint8_t* mask,
                             uint64_t num_elements) {
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  dropout_kernel_half<<<grid_dim, block_dim, 0, stream>>>(input, output, mask,
                                                          seed, num_elements);
}

void LaunchDropoutKernel(CUstream stream, const float* input, float* output,
                         uint8_t* mask, uint64_t num_elements) {
  int block_dim = XLA_CUDA_NUM_THREADS;
  int grid_dim = GET_BLOCKS(num_elements / unroll_factor);
  uint64_t num_threads = grid_dim * block_dim;
  std::pair<int64_t, int64_t> seed =
      CurandContext::Get()->IncrementOffset(num_elements, num_threads);
  dropout_kernel<<<grid_dim, block_dim, 0, stream>>>(input, output, mask, seed,
                                                     num_elements);
}

}  // namespace xla_custom_cuda_ops
