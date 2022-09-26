licenses(["notice"])  # Apache 2.0

load("@local_config_cuda//cuda:build_defs.bzl", "cuda_library")

cc_library(
    name = "bernoulli",
    srcs = [
        "bernoulli.cc",
        "cuda_utils.cc",
        "rectangular_lsap.cc"
    ],
    hdrs = [
        "bernoulli.h",
        "cuda_utils.h",
        "rectangular_lsap.h",
        "lsainfo.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":bernoulli_cu",
        "//tensorflow/compiler/xla:shape_util",
        "//tensorflow/compiler/xla/service:custom_call_status",
        "//tensorflow/compiler/xla/service:custom_call_target_registry",
        "//tensorflow/stream_executor/cuda:curand_lib",
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/status:statusor",
    ],
)

cuda_library(
    name = "bernoulli_cu",
    srcs = [
        "bernoulli.cu.cc",
    ],
    hdrs = [
        "bernoulli.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
